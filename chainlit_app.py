# chainlit_app.py
import asyncio
import sys

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import chainlit as cl
from services.graph import run_chat_graph
from services.db_logging import log_feedback

@cl.on_chat_start
async def on_chat_start():
    # Initialize session state for all arguments required by run_chat_graph
    cl.user_session.set("filters", {})
    cl.user_session.set("booking_args", {})
    cl.user_session.set("status_args", {})
    cl.user_session.set("payment_args", {})

    # Send a welcome message
    await cl.Message(
        content="Welcome to the AI Hotel Booking Assistant! How can I help you today?"
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    # Retrieve current session state
    filters = cl.user_session.get("filters", {})
    booking_args = cl.user_session.get("booking_args", {})
    status_args = cl.user_session.get("status_args", {})
    payment_args = cl.user_session.get("payment_args", {})

    # Disable streaming for the underlying graph request
    filters["stream"] = False

    # Execute the LangGraph routing
    result = await run_chat_graph(
        message=message.content,
        filters=filters,
        booking_args=booking_args,
        status_args=status_args,
        payment_args=payment_args,
    )

    # Persist any updated filters and arguments back into the session state
    cl.user_session.set("filters", result.get("filters", {}))
    cl.user_session.set("booking_args", result.get("booking_args", {}))
    cl.user_session.set("status_args", result.get("status_args", {}))
    cl.user_session.set("payment_args", result.get("payment_args", {}))

    # Output the bot's reply with feedback buttons
    reply = result.get("reply", "Sorry, I didn't understand that.")

    actions = [
        cl.Action(name="thumbs_up", payload={"rating": "positive"}, label="👍"),
        cl.Action(name="thumbs_down", payload={"rating": "negative"}, label="👎"),
    ]

    msg = cl.Message(content=reply, actions=actions)
    await msg.send()

    # Store last exchange for feedback logging
    cl.user_session.set("last_user_msg", message.content)
    cl.user_session.set("last_bot_reply", reply)


@cl.action_callback("thumbs_up")
async def on_thumbs_up(action: cl.Action):
    user_msg = cl.user_session.get("last_user_msg", "")
    bot_reply = cl.user_session.get("last_bot_reply", "")
    log_feedback(user_msg, bot_reply, rating="positive")
    await cl.Message(content="Thanks for the feedback!").send()


@cl.action_callback("thumbs_down")
async def on_thumbs_down(action: cl.Action):
    user_msg = cl.user_session.get("last_user_msg", "")
    bot_reply = cl.user_session.get("last_bot_reply", "")
    log_feedback(user_msg, bot_reply, rating="negative")
    await cl.Message(content="Thanks for the feedback. We'll work on improving!").send()
