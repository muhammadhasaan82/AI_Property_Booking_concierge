# route/mobile.py - Mobile API endpoints
from typing import Any, Dict, Optional, List
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from services.graph import run_chat_graph
from services.search import property_search
from services.booking import create_booking, update_booking_status, get_booking_status
from services.faq import faq_lookup

router = APIRouter()

# ==================== MOBILE AUTHENTICATION ====================

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
    # TODO: Implement actual authentication logic
    # For now, return a mock response
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

# ==================== MOBILE PROPERTIES ====================

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
        
        # Transform results for mobile format
        mobile_results = []
        for prop in results[:20]:  # Limit to 20 results for mobile
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
    # TODO: Implement property details lookup
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

# ==================== MOBILE BOOKING ====================

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
    # TODO: Implement user bookings lookup
    return {
        "success": True,
        "bookings": [],
        "message": "User bookings retrieved"
    }

# ==================== MOBILE CHAT ====================

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
        result = await run_chat_graph(
            message=request.message,
            filters=None,
            booking_args=None,
            status_args=None,
            payment_args=None
        )
        
        return MobileChatResponse(
            success=True,
            response=result.get("reply", "I'm here to help!"),
            session_id=request.session_id,
            suggestions=["Book a property", "Check booking status", "Ask about policies"]
        )
    except Exception as e:
        return MobileChatResponse(
            success=False,
            response=f"Sorry, I encountered an error: {str(e)}"
        )

# ==================== MOBILE FAQ ====================

@router.get("/mobile/faq")
async def mobile_faq(question: str = Query(..., min_length=2)):
    """
    Mobile FAQ endpoint with simplified response
    """
    try:
        answer = faq_lookup(question)
        return {
            "success": True,
            "question": question,
            "answer": answer or "I don't have information about that. Please contact support.",
            "helpful": answer is not None
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"FAQ lookup failed: {str(e)}"
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

# ==================== MOBILE PROFILE ====================

@router.get("/mobile/user/{user_id}/profile")
async def mobile_get_profile(user_id: str):
    """
    Get user profile for mobile
    """
    # TODO: Implement user profile lookup
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
    # TODO: Implement profile update
    return {
        "success": True,
        "message": "Profile updated successfully"
    }

# ==================== MOBILE NOTIFICATIONS ====================

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

# ==================== MOBILE HEALTH ====================

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
