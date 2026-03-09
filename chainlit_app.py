# chainlit_app.py
import asyncio
import sys

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import chainlit as cl
from services.graph import run_chat_graph
from services.db_logging import log_feedback
from services.state_keys import SK


@cl.on_chat_start
async def on_chat_start():
    # Initialize session state for all arguments required by run_chat_graph
    cl.user_session.set("filters", {})
    cl.user_session.set("booking_args", {})
    cl.user_session.set("status_args", {})
    cl.user_session.set("payment_args", {})

    # Send a styled welcome message
    welcome_msg = """# AI Booking Concierge

Welcome! I'm your personal booking assistant.

I can help you:
- **Find properties** in your preferred city
- **Book stays** with your chosen dates and amenities
- **Manage reservations** - check status, modify, or cancel

What would you like to do today?
"""
    await cl.Message(content=welcome_msg).send()


async def stream_callback(token: str, msg: cl.Message):
    """Async callback for streaming LLM tokens to the user."""
    await msg.stream_token(token)


@cl.on_message
async def on_message(message: cl.Message):
    # Retrieve current session state
    filters = cl.user_session.get("filters", {})
    booking_args = cl.user_session.get("booking_args", {})
    status_args = cl.user_session.get("status_args", {})
    payment_args = cl.user_session.get("payment_args", {})

    # Enable streaming for the underlying graph request
    filters["stream"] = True
    msg = cl.Message(content="")
    await msg.send()
    filters["stream_callback"] = lambda token: stream_callback(token, msg)

    # Execute the LangGraph routing
    result = await run_chat_graph(
        message=message.content,
        filters=filters,
        booking_args=booking_args,
        status_args=status_args,
        payment_args=payment_args,
    )

    # Persist updated states
    cl.user_session.set("filters", result.get("filters", {}))
    cl.user_session.set("booking_args", result.get("booking_args", {}))
    cl.user_session.set("status_args", result.get("status_args", {}))
    cl.user_session.set("payment_args", result.get("payment_args", {}))

    reply = result.get("reply", "Sorry, I didn't understand that.")

    # Visual emphasis for active flows
    active_filters = result.get("filters", {})
    is_active_flow = (
        active_filters.get(SK.awaiting_property_type_choice) or
        active_filters.get(SK.awaiting_selection_confirm) or
        active_filters.get(SK.awaiting_city_selection) or
        active_filters.get(SK.awaiting_field) or
        active_filters.get(SK.awaiting_unavailable_city_choice)
    )

    if is_active_flow and reply:
        lines = reply.split('\n')
        formatted_lines = []
        for line in lines:
            if '?' in line or 'please' in line.lower() or 'reply' in line.lower():
                formatted_lines.append(f"**{line}**")
            else:
                formatted_lines.append(line)
        reply = '\n'.join(formatted_lines)

    msg.content = reply

    # Action buttons
    actions = [
        cl.Action(name="thumbs_up", value="positive", label="👍"),
        cl.Action(name="thumbs_down", value="negative", label="👎"),
    ]
    msg.actions = actions

    await msg.update()

    cl.user_session.set("last_user_msg", message.content)
    cl.user_session.set("last_bot_reply", reply)
    cl.user_session.set("last_msg_id", msg.id)


@cl.action_callback("thumbs_up")
async def on_thumbs_up(action: cl.Action):
    user_msg = cl.user_session.get("last_user_msg", "")
    bot_reply = cl.user_session.get("last_bot_reply", "")
    log_feedback(user_msg, bot_reply, rating="positive")

    last_msg_id = cl.user_session.get("last_msg_id")
    if last_msg_id:
        try:
            msg = cl.Message(id=last_msg_id)
            msg.content = f"{bot_reply}\n\n---\n*Thanks for the positive feedback! 🚀*"
            msg.actions = []  
            await msg.update()
        except Exception:
            await cl.Message(content="Thanks for the feedback! 🚀").send()
    else:
        await cl.Message(content="Thanks for the feedback! 🚀").send()


@cl.action_callback("thumbs_down")
async def on_thumbs_down(action: cl.Action):
    user_msg = cl.user_session.get("last_user_msg", "")
    bot_reply = cl.user_session.get("last_bot_reply", "")
    log_feedback(user_msg, bot_reply, rating="negative")

    last_msg_id = cl.user_session.get("last_msg_id")
    if last_msg_id:
        try:
            msg = cl.Message(id=last_msg_id)
            msg.content = f"{bot_reply}\n\n---\n*Thanks for the feedback! We'll work on improving.*"
            msg.actions = [] 
            await msg.update()
        except Exception:
            await cl.Message(content="Thanks for the feedback. We'll work on improving!").send()
    else:
        await cl.Message(content="Thanks for the feedback. We'll work on improving!").send()
