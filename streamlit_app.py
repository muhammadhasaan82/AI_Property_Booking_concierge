#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import asyncio
import os
import subprocess
import sys
from typing import Dict, Any

# Initialize Supabase environment variables if not set
def init_supabase_env():
    """Initialize Supabase environment variables from local development setup."""
    if not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_KEY"):
        try:
            # Get Supabase local development URLs
            res = subprocess.run(["supabase", "status", "-o", "env"], 
                               check=False, capture_output=True, text=True)
            for line in (res.stdout or "").splitlines():
                if line.startswith("API_URL="):
                    os.environ["SUPABASE_URL"] = line.split("=", 1)[1].strip()
                elif line.startswith("SERVICE_ROLE_KEY="):
                    os.environ["SUPABASE_KEY"] = line.split("=", 1)[1].strip()
            
            # Initialize database schema
            try:
                from services import db_setup as _db
                _db.init_schema(None)
                return True
            except Exception as e:
                st.error(f"Database schema initialization failed: {e}")
                return False
        except Exception as e:
            st.warning(f"Could not initialize Supabase: {e}")
            return False
    return True

# Initialize Supabase before importing chatbot functionality
supabase_ready = init_supabase_env()

# Import the chatbot functionality
from services.graph import run_chat_graph

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "filters" not in st.session_state:
    st.session_state.filters = {
        "budget": None,
        "beds": None,
        "location": None,
        "amenities": [],
        "locale": "en",
        "stream": False,  # Disable streaming for Streamlit
        "stream_callback": None,
    }

# Page config
st.set_page_config(
    page_title="AgenticRAG Property Assistant",
    page_icon="🏠",
    layout="centered"
)

# Title
st.title("🏠 Property Assistant")
st.markdown("Your AI-powered property booking assistant")

# Show Supabase status
if supabase_ready:
    st.success("✅ Database connected - Chat history and bookings will be saved!")
    # Show database stats
    try:
        from services import db_setup as _db
        stats = _db.verify(None)
        if stats.get("ok"):
            st.info(f"📊 Database: {stats.get('users', 0)} users, {stats.get('bookings', 0)} bookings, {stats.get('chat_history', 0)} chat messages")
    except:
        pass
else:
    st.warning("⚠️ Database not connected - Chat history and bookings will not be saved")

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me about properties, bookings, or anything else!"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Run the chat graph
            result = asyncio.run(run_chat_graph(
                message=prompt,
                filters=st.session_state.filters,
                booking_args={},
                status_args={},
                payment_args={},
            ))

            # Update filters if changed
            if result.get("filters"):
                st.session_state.filters.update(result["filters"])

            # Display response
            response = result.get("reply", "I'm sorry, I couldn't process that request.")
            st.markdown(response)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar with options
with st.sidebar:
    st.header("Settings")
    
    # Budget filter
    budget = st.number_input("Budget ($/night)", min_value=0, value=0, step=50)
    if budget > 0:
        st.session_state.filters["budget"] = budget
    else:
        st.session_state.filters["budget"] = None
    
    # Beds filter
    beds = st.number_input("Minimum bedrooms", min_value=0, value=0, step=1)
    if beds > 0:
        st.session_state.filters["beds"] = beds
    else:
        st.session_state.filters["beds"] = None
    
    # Location filter
    location = st.text_input("Location (city)")
    st.session_state.filters["location"] = location if location else None
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    # Show current filters
    st.header("Current Filters")
    active_filters = {k: v for k, v in st.session_state.filters.items() 
                     if v is not None and v != [] and k not in ["stream", "stream_callback", "locale"]}
    if active_filters:
        for key, value in active_filters.items():
            st.write(f"**{key.title()}:** {value}")
    else:
        st.write("No active filters")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Powered by AgenticRAG")
