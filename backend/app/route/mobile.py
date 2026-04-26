from typing import Any, Dict, Optional, List
from uuid import uuid4
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from app.services.adk_runner import run_adk_turn
from app.components.search import property_search
from app.services.booking import create_booking, update_booking_status, get_booking_status

router = APIRouter()

class MobileLoginRequest(BaseModel):
    email: str
    password: str
    device_id: Optional[str] = None

class MobileLoginResponse(BaseModel):
    success: bool
    user_id: Optional[str] = None
    token: Optional[str] = None
    message: str

@router.post("/mobile/auth/login", response_model=MobileLoginResponse)
async def mobile_login(request: MobileLoginRequest):
    """
    Mobile user authentication endpoint
    """
    return MobileLoginResponse(
        success=True,
        user_id="mobile_user_123",
        token="mock_jwt_token_123",
        message="Login successful"
    )

@router.post("/mobile/auth/logout")
async def mobile_logout(user_id: str):
    """
    Mobile user logout endpoint
    """
    return {"success": True, "message": "Logout successful"}

class MobilePropertySearchRequest(BaseModel):
    query: str = ""
    location: Optional[str] = None
    budget_min: Optional[float] = None
    budget_max: Optional[float] = None
    guests: Optional[int] = None
    check_in: Optional[str] = None
    check_out: Optional[str] = None
    amenities: Optional[List[str]] = None

class MobilePropertyResponse(BaseModel):
    id: str
    title: str
    location: str
    price_per_night: float
    rating: Optional[float] = None
    image_url: Optional[str] = None
    amenities: List[str] = []
    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None

@router.post("/mobile/properties/search")
async def mobile_search_properties(request: MobilePropertySearchRequest):
    """
    Mobile-optimized property search
    """
    try:
        results = property_search(
            query_text=request.query,
            budget=request.budget_max,
            amenities=request.amenities,
            location=request.location,
            beds=request.guests
        )

        mobile_results = []
        for prop in results[:20]:
            mobile_results.append(MobilePropertyResponse(
                id=str(prop.get("id", "")),
                title=prop.get("title", "Property"),
                location=prop.get("location", ""),
                price_per_night=float(prop.get("price", 0)),
                rating=prop.get("rating"),
                image_url=prop.get("image_url"),
                amenities=prop.get("amenities", []),
                bedrooms=prop.get("bedrooms"),
                bathrooms=prop.get("bathrooms")
            ))
        
        return {
            "success": True,
            "results": mobile_results,
            "total": len(mobile_results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.get("/mobile/properties/{property_id}")
async def mobile_get_property(property_id: str):
    """
    Get detailed property information for mobile
    """
    return {
        "success": True,
        "property": {
            "id": property_id,
            "title": "Sample Property",
            "description": "Beautiful property for your stay",
            "price_per_night": 150.0,
            "location": "Sample Location",
            "amenities": ["wifi", "pool", "kitchen"],
            "images": [],
            "rating": 4.5,
            "reviews_count": 25
        }
    }

class MobileBookingRequest(BaseModel):
    user_id: str
    property_id: str
    check_in: str
    check_out: str
    guests: int
    guest_name: str
    guest_email: str
    guest_phone: str
    special_requests: Optional[str] = None

class MobileBookingResponse(BaseModel):
    success: bool
    booking_id: Optional[str] = None
    total_amount: Optional[float] = None
    payment_url: Optional[str] = None
    message: str

@router.post("/mobile/booking/create", response_model=MobileBookingResponse)
async def mobile_create_booking(request: MobileBookingRequest):
    """
    Mobile booking creation endpoint
    """
    try:
        booking_data = {
            "user_id": request.user_id,
            "property_id": request.property_id,
            "check_in": request.check_in,
            "check_out": request.check_out,
            "guests": request.guests,
            "phone": request.guest_phone
        }
        
        result = create_booking(booking_data)
        
        if result.get("success"):
            return MobileBookingResponse(
                success=True,
                booking_id=result.get("booking_id"),
                total_amount=result.get("total_amount"),
                payment_url=result.get("payment_url"),
                message="Booking created successfully"
            )
        else:
            return MobileBookingResponse(
                success=False,
                message=result.get("error", "Booking failed")
            )
    except Exception as e:
        return MobileBookingResponse(
            success=False,
            message=f"Booking failed: {str(e)}"
        )

@router.get("/mobile/booking/{booking_id}")
async def mobile_get_booking(booking_id: str):
    """
    Get booking details for mobile
    """
    try:
        result = get_booking_status(booking_id)
        return {
            "success": True,
            "booking": result
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Booking not found: {str(e)}")

@router.get("/mobile/user/{user_id}/bookings")
async def mobile_get_user_bookings(user_id: str):
    """
    Get all bookings for a mobile user
    """
    return {
        "success": True,
        "bookings": [],
        "message": "User bookings retrieved"
    }

class MobileChatRequest(BaseModel):
    user_id: str
    message: str
    session_id: Optional[str] = None

class MobileChatResponse(BaseModel):
    success: bool
    response: str
    session_id: Optional[str] = None
    suggestions: Optional[List[str]] = None

@router.post("/mobile/chat", response_model=MobileChatResponse)
async def mobile_chat(request: MobileChatRequest):
    """
    Mobile chat endpoint with simplified response
    """
    try:
        user_id = request.user_id or "mobile_user"
        session_id = request.session_id or str(uuid4())

        chunks = []
        async for chunk in run_adk_turn(
            user_id=user_id,
            session_id=session_id,
            message=request.message,
        ):
            chunks.append(chunk)

        return MobileChatResponse(
            success=True,
            response="".join(chunks),
            session_id=session_id,
            suggestions=["Book a property", "Check booking status", "Ask about policies"]
        )
    except Exception as e:
        return MobileChatResponse(
            success=False,
            response=f"Sorry, I encountered an error: {str(e)}"
        )

@router.get("/mobile/faq")
async def mobile_faq(request: dict):
    """Mobile FAQ goes through the full ADK pipeline (CAG -> RAG -> fallback)."""
    question = (request.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    session_id  = request.get("session_id") or f"mobile-{uuid4()}"
    user_id = request.get("user_id") or "mobile-anon"

    result = await run_adk_turn(
        user_message=question,
        session_id=session_id,
        user_id=user_id,
    )

    return{
        "success":True,
        "answer": result.get("reply") or result.get("final_reply") or "",
        "session_id": session_id
    }

@router.get("/mobile/faq/categories")
async def mobile_faq_categories():
    """
    Get FAQ categories for mobile
    """
    return {
        "success": True,
        "categories": [
            {"id": "booking", "name": "Booking", "count": 5},
            {"id": "payment", "name": "Payment", "count": 3},
            {"id": "cancellation", "name": "Cancellation", "count": 4},
            {"id": "checkin", "name": "Check-in/Check-out", "count": 3},
            {"id": "amenities", "name": "Amenities", "count": 2}
        ]
    }


@router.get("/mobile/user/{user_id}/profile")
async def mobile_get_profile(user_id: str):
    """
    Get user profile for mobile
    """
    return {
        "success": True,
        "profile": {
            "user_id": user_id,
            "name": "Mobile User",
            "email": "user@example.com",
            "phone": "+1234567890",
            "preferences": {
                "notifications": True,
                "language": "en"
            }
        }
    }

@router.put("/mobile/user/{user_id}/profile")
async def mobile_update_profile(user_id: str, profile_data: dict):
    """
    Update user profile for mobile
    """
    return {
        "success": True,
        "message": "Profile updated successfully"
    }

@router.get("/mobile/user/{user_id}/notifications")
async def mobile_get_notifications(user_id: str):
    """
    Get user notifications for mobile
    """
    return {
        "success": True,
        "notifications": [
            {
                "id": "1",
                "title": "Booking Confirmed",
                "message": "Your booking has been confirmed",
                "type": "booking",
                "read": False,
                "created_at": "2025-01-08T10:00:00Z"
            }
        ]
    }

@router.post("/mobile/user/{user_id}/notifications/{notification_id}/read")
async def mobile_mark_notification_read(user_id: str, notification_id: str):
    """
    Mark notification as read
    """
    return {
        "success": True,
        "message": "Notification marked as read"
    }

@router.get("/mobile/health")
async def mobile_health():
    """
    Mobile-specific health check
    """
    return {
        "success": True,
        "status": "healthy",
        "version": "1.0.0",
        "mobile_api": True,
        "timestamp": "2025-01-08T10:00:00Z"
    }

