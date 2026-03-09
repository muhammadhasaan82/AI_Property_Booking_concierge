# chainlit_app.py
import asyncio
import os
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import chainlit as cl

from services.db_logging import log_feedback
from services.graph import run_chat_graph
from services.state_keys import SK

WELCOME_MESSAGE = """# AI Hotel Concierge

Welcome! I'm your personal booking assistant.

I can help you:
- **Find properties** in your preferred city
- **Book stays** with your chosen dates and amenities
- **Manage reservations** - check status, modify, or cancel

What would you like to do today?
"""

AUTH_ENABLED = bool(os.environ.get("CHAINLIT_AUTH_SECRET"))


async def _send_toast(message: str, toast_type: str) -> None:
    try:
        await cl.context.emitter.send_toast(message=message, type=toast_type)
    except Exception:
        pass


def _make_stream_callback(msg: cl.Message):
    def _callback(token: str) -> None:
        asyncio.create_task(msg.stream_token(token))

    return _callback


async def _rename_thread(message: cl.Message, question: str) -> None:
    if not question:
        return

    short_q = question[:15].rstrip()
    if len(question) > 15:
        short_q += "..."

    try:
        data_layer = getattr(cl, "data_layer", None)
        thread_id = getattr(message, "thread_id", None)
        if data_layer and thread_id:
            await data_layer.update_thread(thread_id, name=f"Booking: {short_q}")
    except Exception:
        pass


async def _update_feedback_message(
    bot_reply: str,
    acknowledgement: str,
    fallback_message: str,
) -> None:
    last_msg_id = cl.user_session.get("last_msg_id")
    if last_msg_id:
        try:
            msg = cl.Message(id=last_msg_id)
            msg.content = f"{bot_reply}\n\n---\n*{acknowledgement}*"
            msg.actions = []
            await msg.update()
            return
        except Exception:
            pass

    await cl.Message(content=fallback_message).send()


if AUTH_ENABLED:
    @cl.password_auth_callback
    def auth_callback(username: str, password: str):
        """Accept any username/password for now; swap in real auth later."""
        return cl.User(identifier=username)


@cl.on_chat_resume
async def on_chat_resume(thread):
    """Restore minimal session context when a user resumes an old thread."""
    if isinstance(thread, dict):
        past_thread_id = thread.get("id")
        if past_thread_id:
            cl.user_session.set("past_thread_id", past_thread_id)

    await _send_toast(
        "Welcome back! Your booking context has been restored.",
        "success",
    )


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("filters", {})
    cl.user_session.set("booking_args", {})
    cl.user_session.set("status_args", {})
    cl.user_session.set("payment_args", {})

    await _send_toast("AI Concierge initialized and ready.", "info")
    await cl.Message(content=WELCOME_MESSAGE).send()


@cl.on_message
async def on_message(message: cl.Message):
    filters = dict(cl.user_session.get("filters", {}) or {})
    booking_args = dict(cl.user_session.get("booking_args", {}) or {})
    status_args = dict(cl.user_session.get("status_args", {}) or {})
    payment_args = dict(cl.user_session.get("payment_args", {}) or {})

    await _rename_thread(message, (message.content or "").strip())

    filters["stream"] = True
    msg = cl.Message(content="")
    await msg.send()
    filters["stream_callback"] = _make_stream_callback(msg)

    result = await run_chat_graph(
        message=message.content,
        filters=filters,
        booking_args=booking_args,
        status_args=status_args,
        payment_args=payment_args,
    )
    result = result or {}

    cl.user_session.set("filters", result.get("filters", {}))
    cl.user_session.set("booking_args", result.get("booking_args", {}))
    cl.user_session.set("status_args", result.get("status_args", {}))
    cl.user_session.set("payment_args", result.get("payment_args", {}))

    reply = result.get("reply", "Sorry, I didn't understand that.")
    active_filters = result.get("filters", {})
    is_active_flow = (
        active_filters.get(SK.awaiting_property_type_choice)
        or active_filters.get(SK.awaiting_selection_confirm)
        or active_filters.get(SK.awaiting_city_selection)
        or active_filters.get(SK.awaiting_field)
        or active_filters.get(SK.awaiting_unavailable_city_choice)
    )

    if is_active_flow and reply:
        formatted_lines = []
        for line in reply.split("\n"):
            if "?" in line or "please" in line.lower() or "reply" in line.lower():
                formatted_lines.append(f"**{line}**")
            else:
                formatted_lines.append(line)
        reply = "\n".join(formatted_lines)

    msg.content = reply
    msg.actions = [
        cl.Action(name="thumbs_up", value="positive", label="👍"),
        cl.Action(name="thumbs_down", value="negative", label="👎"),
    ]
    await msg.update()

    cl.user_session.set("last_user_msg", message.content)
    cl.user_session.set("last_bot_reply", reply)
    cl.user_session.set("last_msg_id", msg.id)


@cl.action_callback("thumbs_up")
async def on_thumbs_up(action: cl.Action):
    user_msg = cl.user_session.get("last_user_msg", "")
    bot_reply = cl.user_session.get("last_bot_reply", "")
    await log_feedback(user_msg, bot_reply, rating="positive")
    await _update_feedback_message(
        bot_reply,
        "Thanks for the positive feedback! 🚀",
        "Thanks for the feedback! 🚀",
    )


@cl.action_callback("thumbs_down")
async def on_thumbs_down(action: cl.Action):
    user_msg = cl.user_session.get("last_user_msg", "")
    bot_reply = cl.user_session.get("last_bot_reply", "")
    await log_feedback(user_msg, bot_reply, rating="negative")
    await _update_feedback_message(
        bot_reply,
        "Thanks for the feedback! We'll work on improving.",
        "Thanks for the feedback. We'll work on improving!",
    )
