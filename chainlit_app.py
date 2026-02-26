# chainlit_app.py
import chainlit as cl
from services.graph import run_chat_graph

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

    # Output the bot's reply directly to the UI
    reply = result.get("reply", "Sorry, I didn't understand that.")
    await cl.Message(content=reply).send()
