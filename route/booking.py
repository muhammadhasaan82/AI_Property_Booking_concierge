# route/booking.py
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from services.booking import create_booking, update_booking_status, get_booking_status

router = APIRouter()

class CreateBookingIn(BaseModel):
    user_id: str
    property_id: str
    check_in: str
    check_out: str
    guests: int = 1
    phone: Optional[str] = None

@router.post("/booking/create")
async def booking_create(body: CreateBookingIn):
    r = await create_booking(body.model_dump())
    return r

class UpdateStatusIn(BaseModel):
    booking_id: str
    current_status: str
    new_status: str  # confirmed | checked_in | checked_out

@router.post("/booking/update-status")
async def booking_update_status(body: UpdateStatusIn):
    r = await update_booking_status(body.booking_id, body.current_status, body.new_status)
    return r

@router.get("/booking/status/{booking_id}")
async def booking_status(booking_id: str):
    r = await get_booking_status(booking_id)
    return r
