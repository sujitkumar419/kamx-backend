"""
KamX Backend — FastAPI
Real auth, users, partner discovery, search, bookings with Razorpay payment,
and commission logic.
Run locally: uvicorn main:app --reload
Deploy: Railway / Render (see DEPLOY.md)
"""
import os
import uuid
import random
import string
import hmac
import hashlib
from datetime import datetime, timedelta
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Depends, status, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from jose import jwt, JWTError
from sqlalchemy import create_engine, Column, String, Float, Boolean, Integer, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./kamx.db")
SECRET_KEY = os.getenv("SECRET_KEY", "CHANGE_THIS_IN_PRODUCTION_" + str(uuid.uuid4()))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 30  # 30 days

# Razorpay — set these as environment variables once you have them.
# Test mode keys look like: rzp_test_xxxxxxxxxxxx
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "")
RAZORPAY_ENABLED = bool(RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET)

razorpay_client = None
if RAZORPAY_ENABLED:
    import razorpay
    razorpay_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))

# Render/Railway give postgres:// — SQLAlchemy needs postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login", auto_error=False)

app = FastAPI(title="KamX API")

# CORS — replace "*" with your real frontend domain once deployed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict to ["https://your-kamx-domain.com"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Service catalog (used for search + validation)
# ---------------------------------------------------------------------------
SERVICES = [
    {"key": "electrician", "label": "Electrician", "base_price": 199},
    {"key": "ac", "label": "AC / Fridge Repair", "base_price": 349},
    {"key": "plumber", "label": "Plumber", "base_price": 249},
    {"key": "cleaning", "label": "Home Cleaning", "base_price": 299},
]
SERVICE_BASE_PRICE = {s["key"]: s["base_price"] for s in SERVICES}
SERVICE_KEYS = {s["key"] for s in SERVICES}
DEFAULT_PLATFORM_COMMISSION_PCT = 15  # fallback only; real bookings use the partner's own rate


# ---------------------------------------------------------------------------
# DB Models
# ---------------------------------------------------------------------------
class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=True, index=True)
    phone = Column(String, unique=True, nullable=True, index=True)
    password_hash = Column(String, nullable=True)
    is_partner = Column(Boolean, default=False)
    partner_commission = Column(Integer, default=15)  # 10-20
    partner_service = Column(String, nullable=True)    # one of SERVICE_KEYS
    partner_city = Column(String, nullable=True)       # free-text city/area for now
    partner_online = Column(Boolean, default=False)    # toggle: accepting jobs right now
    wallet = Column(Float, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


class OtpCode(Base):
    __tablename__ = "otp_codes"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    phone = Column(String, index=True, nullable=False)
    code = Column(String, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    used = Column(Boolean, default=False)


class Booking(Base):
    __tablename__ = "bookings"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    customer_id = Column(String, ForeignKey("users.id"), nullable=False)
    partner_id = Column(String, ForeignKey("users.id"), nullable=True)
    service_key = Column(String, nullable=False)
    price = Column(Float, nullable=False)
    commission_pct = Column(Integer, nullable=False)
    commission_amt = Column(Float, nullable=False)
    worker_amt = Column(Float, nullable=False)
    status = Column(String, default="pending_payment")
    # pending_payment | confirmed | completed | cancelled
    razorpay_order_id = Column(String, nullable=True)
    razorpay_payment_id = Column(String, nullable=True)
    paid = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class SignupEmail(BaseModel):
    name: str
    email: EmailStr
    password: str


class LoginEmail(BaseModel):
    email: EmailStr
    password: str


class RequestOtp(BaseModel):
    phone: str
    name: Optional[str] = None  # required on signup flow


class VerifyOtp(BaseModel):
    phone: str
    code: str
    name: Optional[str] = None


class UserOut(BaseModel):
    id: str
    name: str
    email: Optional[str]
    phone: Optional[str]
    is_partner: bool
    partner_commission: int
    partner_service: Optional[str]
    partner_city: Optional[str]
    partner_online: bool
    wallet: float

    class Config:
        from_attributes = True


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class BecomePartner(BaseModel):
    commission_pct: int
    service_key: str
    city: Optional[str] = None


class UpdateAvailability(BaseModel):
    online: bool


class ServiceOut(BaseModel):
    key: str
    label: str
    base_price: int


class WorkerOut(BaseModel):
    id: str
    name: str
    service_key: str
    city: Optional[str]
    commission_pct: int

    class Config:
        from_attributes = True


class CreateBooking(BaseModel):
    service_key: str
    price: Optional[float] = None
    partner_id: Optional[str] = None  # customer can pick a specific available worker


class BookingOut(BaseModel):
    id: str
    customer_id: str
    partner_id: Optional[str]
    service_key: str
    price: float
    commission_pct: int
    commission_amt: float
    worker_amt: float
    status: str
    paid: bool
    razorpay_order_id: Optional[str]

    class Config:
        from_attributes = True


class VerifyPayment(BaseModel):
    booking_id: str
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------
def create_access_token(user_id: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode({"sub": user_id, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    cred_exc = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    if not token:
        raise cred_exc
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            raise cred_exc
    except JWTError:
        raise cred_exc
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise cred_exc
    return user


def gen_otp() -> str:
    return "".join(random.choices(string.digits, k=6))


# ---------------------------------------------------------------------------
# Routes — Auth
# ---------------------------------------------------------------------------
@app.post("/auth/signup/email", response_model=TokenOut)
def signup_email(payload: SignupEmail, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(400, "Email already registered. Please login.")
    user = User(
        name=payload.name,
        email=payload.email,
        password_hash=pwd_context.hash(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_access_token(user.id)
    return TokenOut(access_token=token, user=UserOut.model_validate(user))


@app.post("/auth/login/email", response_model=TokenOut)
def login_email(payload: LoginEmail, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not user.password_hash or not pwd_context.verify(payload.password, user.password_hash):
        raise HTTPException(401, "Incorrect email or password")
    token = create_access_token(user.id)
    return TokenOut(access_token=token, user=UserOut.model_validate(user))


@app.post("/auth/otp/request")
def request_otp(payload: RequestOtp, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.phone == payload.phone).first()
    is_signup = payload.name is not None
    if is_signup and existing:
        raise HTTPException(400, "This number is already registered. Please login.")
    if not is_signup and not existing:
        raise HTTPException(404, "No account found. Please sign up first.")

    code = gen_otp()
    otp = OtpCode(phone=payload.phone, code=code, expires_at=datetime.utcnow() + timedelta(minutes=10))
    db.add(otp)
    db.commit()

    # TODO: integrate real SMS provider here (Twilio / MSG91). For now we
    # return the code directly so the demo works without an SMS account.
    # In production, REMOVE `"demo_code": code` from the response.
    send_sms_stub(payload.phone, code)
    return {"message": "OTP sent", "demo_code": code}


def send_sms_stub(phone: str, code: str):
    """
    Replace this with a real SMS provider call, e.g. Twilio:

    from twilio.rest import Client
    client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
    client.messages.create(
        body=f"Your KamX OTP is {code}",
        from_=TWILIO_PHONE_NUMBER,
        to=phone,
    )
    """
    print(f"[SMS STUB] Would send OTP {code} to {phone}")


@app.post("/auth/otp/verify", response_model=TokenOut)
def verify_otp(payload: VerifyOtp, db: Session = Depends(get_db)):
    otp = (
        db.query(OtpCode)
        .filter(OtpCode.phone == payload.phone, OtpCode.code == payload.code, OtpCode.used == False)
        .order_by(OtpCode.expires_at.desc())
        .first()
    )
    if not otp or otp.expires_at < datetime.utcnow():
        raise HTTPException(400, "Invalid or expired OTP")
    otp.used = True
    db.commit()

    user = db.query(User).filter(User.phone == payload.phone).first()
    if not user:
        if not payload.name:
            raise HTTPException(400, "Name is required for signup")
        user = User(name=payload.name, phone=payload.phone)
        db.add(user)
        db.commit()
        db.refresh(user)

    token = create_access_token(user.id)
    return TokenOut(access_token=token, user=UserOut.model_validate(user))


@app.get("/auth/me", response_model=UserOut)
def get_me(current_user: User = Depends(get_current_user)):
    return UserOut.model_validate(current_user)


# ---------------------------------------------------------------------------
# Routes — Services & Search
# ---------------------------------------------------------------------------
@app.get("/services", response_model=List[ServiceOut])
def list_services(q: Optional[str] = Query(None, description="search text")):
    """
    Returns the service catalog, optionally filtered by a search query
    matched against the service label (case-insensitive substring match).
    This powers the home screen search bar.
    """
    results = SERVICES
    if q:
        q_lower = q.strip().lower()
        results = [s for s in SERVICES if q_lower in s["label"].lower() or q_lower in s["key"]]
    return results


# ---------------------------------------------------------------------------
# Routes — Partner profile & discovery
# ---------------------------------------------------------------------------
@app.post("/partner/activate", response_model=UserOut)
def activate_partner(payload: BecomePartner, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not (10 <= payload.commission_pct <= 20):
        raise HTTPException(400, "Commission must be between 10 and 20 percent")
    if payload.service_key not in SERVICE_KEYS:
        raise HTTPException(400, f"Unknown service. Choose one of: {', '.join(SERVICE_KEYS)}")
    current_user.is_partner = True
    current_user.partner_commission = payload.commission_pct
    current_user.partner_service = payload.service_key
    current_user.partner_city = payload.city
    current_user.partner_online = True  # go live immediately on activation
    db.commit()
    db.refresh(current_user)
    return UserOut.model_validate(current_user)


@app.post("/partner/commission", response_model=UserOut)
def update_commission(payload: BecomePartner, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user.is_partner:
        raise HTTPException(400, "You are not a partner yet")
    if not (10 <= payload.commission_pct <= 20):
        raise HTTPException(400, "Commission must be between 10 and 20 percent")
    current_user.partner_commission = payload.commission_pct
    if payload.service_key in SERVICE_KEYS:
        current_user.partner_service = payload.service_key
    if payload.city is not None:
        current_user.partner_city = payload.city
    db.commit()
    db.refresh(current_user)
    return UserOut.model_validate(current_user)


@app.post("/partner/availability", response_model=UserOut)
def set_availability(payload: UpdateAvailability, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user.is_partner:
        raise HTTPException(400, "You are not a partner yet")
    current_user.partner_online = payload.online
    db.commit()
    db.refresh(current_user)
    return UserOut.model_validate(current_user)


@app.get("/partner/bookings", response_model=List[BookingOut])
def partner_bookings(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user.is_partner:
        raise HTTPException(400, "You are not a partner yet")
    bookings = (
        db.query(Booking)
        .filter(Booking.partner_id == current_user.id)
        .order_by(Booking.created_at.desc())
        .limit(50)
        .all()
    )
    return [BookingOut.model_validate(b) for b in bookings]


@app.get("/workers", response_model=List[WorkerOut])
def find_workers(
    service_key: str = Query(..., description="which service the customer wants"),
    city: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """
    Real worker discovery — only returns partners who are:
      - registered for this exact service
      - currently toggled online
    No fake/placeholder names are ever returned. If the list is empty,
    the frontend should show "no worker available right now" honestly.
    """
    if service_key not in SERVICE_KEYS:
        raise HTTPException(400, "Unknown service_key")
    query = db.query(User).filter(
        User.is_partner == True,
        User.partner_online == True,
        User.partner_service == service_key,
    )
    if city:
        query = query.filter(User.partner_city.ilike(f"%{city}%"))
    workers = query.limit(20).all()
    return [WorkerOut.model_validate(w) for w in workers]


# ---------------------------------------------------------------------------
# Routes — Bookings + Razorpay payment
# ---------------------------------------------------------------------------
@app.get("/payments/config")
def payment_config():
    """Frontend calls this to know whether real payment is wired up,
    and to get the public Key ID needed to open the Razorpay checkout."""
    return {"razorpay_enabled": RAZORPAY_ENABLED, "key_id": RAZORPAY_KEY_ID if RAZORPAY_ENABLED else None}


@app.post("/bookings", response_model=BookingOut)
def create_booking(payload: CreateBooking, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if payload.service_key not in SERVICE_KEYS:
        raise HTTPException(400, "Unknown service_key")

    partner = None
    if payload.partner_id:
        partner = (
            db.query(User)
            .filter(User.id == payload.partner_id, User.is_partner == True, User.partner_online == True)
            .first()
        )
        if not partner:
            raise HTTPException(400, "Selected worker is no longer available")

    price = payload.price or SERVICE_BASE_PRICE.get(payload.service_key, 250)
    commission_pct = partner.partner_commission if partner else DEFAULT_PLATFORM_COMMISSION_PCT
    commission_amt = round(price * commission_pct / 100, 2)
    worker_amt = round(price - commission_amt, 2)

    booking = Booking(
        customer_id=current_user.id,
        partner_id=partner.id if partner else None,
        service_key=payload.service_key,
        price=price,
        commission_pct=commission_pct,
        commission_amt=commission_amt,
        worker_amt=worker_amt,
        status="pending_payment",
        paid=False,
    )

    if RAZORPAY_ENABLED:
        order = razorpay_client.order.create({
            "amount": int(price * 100),  # Razorpay wants paise, not rupees
            "currency": "INR",
            "payment_capture": 1,
            "notes": {"booking_service": payload.service_key},
        })
        booking.razorpay_order_id = order["id"]
    else:
        # No payment gateway configured yet — fall back to auto-confirm so
        # the rest of the app stays testable. Remove this branch once
        # RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET are set in production.
        booking.status = "confirmed"
        booking.paid = True

    db.add(booking)
    db.commit()
    db.refresh(booking)
    return BookingOut.model_validate(booking)


@app.post("/bookings/verify-payment", response_model=BookingOut)
def verify_payment(payload: VerifyPayment, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not RAZORPAY_ENABLED:
        raise HTTPException(400, "Payments are not configured on this server")

    booking = db.query(Booking).filter(Booking.id == payload.booking_id, Booking.customer_id == current_user.id).first()
    if not booking:
        raise HTTPException(404, "Booking not found")
    if booking.razorpay_order_id != payload.razorpay_order_id:
        raise HTTPException(400, "Order mismatch")

    # Verify Razorpay's HMAC signature ourselves (don't trust the client blindly)
    body = f"{payload.razorpay_order_id}|{payload.razorpay_payment_id}"
    expected_signature = hmac.new(
        RAZORPAY_KEY_SECRET.encode(), body.encode(), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(expected_signature, payload.razorpay_signature):
        raise HTTPException(400, "Payment signature verification failed")

    booking.razorpay_payment_id = payload.razorpay_payment_id
    booking.paid = True
    booking.status = "confirmed"
    db.commit()
    db.refresh(booking)
    return BookingOut.model_validate(booking)


@app.get("/bookings/mine", response_model=List[BookingOut])
def my_bookings(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    bookings = (
        db.query(Booking)
        .filter(Booking.customer_id == current_user.id)
        .order_by(Booking.created_at.desc())
        .all()
    )
    return [BookingOut.model_validate(b) for b in bookings]


@app.post("/bookings/{booking_id}/accept", response_model=BookingOut)
def accept_booking(booking_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user.is_partner:
        raise HTTPException(400, "Only partners can accept bookings")
    booking = db.query(Booking).filter(Booking.id == booking_id).first()
    if not booking:
        raise HTTPException(404, "Booking not found")
    if booking.partner_id and booking.partner_id != current_user.id:
        raise HTTPException(400, "Booking already assigned to another partner")
    if not booking.paid:
        raise HTTPException(400, "Booking has not been paid for yet")

    booking.partner_id = current_user.id
    booking.commission_pct = current_user.partner_commission
    booking.commission_amt = round(booking.price * current_user.partner_commission / 100, 2)
    booking.worker_amt = round(booking.price - booking.commission_amt, 2)
    db.commit()
    db.refresh(booking)
    return BookingOut.model_validate(booking)


@app.get("/health")
def health():
    return {"status": "ok", "razorpay_enabled": RAZORPAY_ENABLED}


# ---------------------------------------------------------------------------
# Routes — Admin Dashboard (simple fixed-password protected)
# ---------------------------------------------------------------------------
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "kamx-admin-2026")


class AdminLogin(BaseModel):
    password: str


def check_admin(x_admin_password: Optional[str] = None):
    if x_admin_password != ADMIN_PASSWORD:
        raise HTTPException(401, "Invalid admin password")
    return True


@app.post("/admin/login")
def admin_login(payload: AdminLogin):
    if payload.password != ADMIN_PASSWORD:
        raise HTTPException(401, "Incorrect password")
    return {"ok": True}


@app.get("/admin/stats")
def admin_stats(x_admin_password: str = Header(default=""), db: Session = Depends(get_db)):
    check_admin(x_admin_password)
    total_users = db.query(User).count()
    total_partners = db.query(User).filter(User.is_partner == True).count()
    online_partners = db.query(User).filter(User.is_partner == True, User.partner_online == True).count()
    total_bookings = db.query(Booking).count()
    paid_bookings = db.query(Booking).filter(Booking.paid == True).all()
    total_revenue = sum(b.price for b in paid_bookings)
    total_commission = sum(b.commission_amt for b in paid_bookings)
    return {
        "total_users": total_users,
        "total_partners": total_partners,
        "online_partners": online_partners,
        "total_bookings": total_bookings,
        "paid_bookings": len(paid_bookings),
        "total_revenue": round(total_revenue, 2),
        "total_commission_earned": round(total_commission, 2),
    }


@app.get("/admin/users")
def admin_users(x_admin_password: str = Header(default=""), db: Session = Depends(get_db)):
    check_admin(x_admin_password)
    users = db.query(User).order_by(User.created_at.desc()).all()
    return [
        {
            "id": u.id,
            "name": u.name,
            "email": u.email,
            "phone": u.phone,
            "is_partner": u.is_partner,
            "partner_service": u.partner_service,
            "partner_city": u.partner_city,
            "partner_online": u.partner_online,
            "partner_commission": u.partner_commission,
            "wallet": u.wallet,
            "created_at": u.created_at.isoformat() if u.created_at else None,
        }
        for u in users
    ]


@app.get("/admin/bookings")
def admin_bookings(x_admin_password: str = Header(default=""), db: Session = Depends(get_db)):
    check_admin(x_admin_password)
    bookings = db.query(Booking).order_by(Booking.created_at.desc()).limit(200).all()
    result = []
    for b in bookings:
        customer = db.query(User).filter(User.id == b.customer_id).first()
        partner = db.query(User).filter(User.id == b.partner_id).first() if b.partner_id else None
        result.append({
            "id": b.id,
            "customer_name": customer.name if customer else "Unknown",
            "partner_name": partner.name if partner else None,
            "service_key": b.service_key,
            "price": b.price,
            "commission_pct": b.commission_pct,
            "commission_amt": b.commission_amt,
            "worker_amt": b.worker_amt,
            "status": b.status,
            "paid": b.paid,
            "created_at": b.created_at.isoformat() if b.created_at else None,
        })
    return result

