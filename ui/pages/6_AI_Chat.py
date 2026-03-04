"""AI Chat — interactive chat powered by NVIDIA NIM with tool calling."""

import sys
from pathlib import Path
_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import streamlit as st
from ui.utils.state import init_state
from ui.components.sidebar import render_sidebar
from ui.utils.api_client import get_api_client

init_state()
render_sidebar()

st.header("AI Chat")
st.caption("Renewable energy analysis assistant powered by NVIDIA NIM")

api = get_api_client()

# Check LLM status
llm_info = api.llm_status()
if llm_info and llm_info.get("available"):
    provider = llm_info.get("provider", "unknown")
    model = llm_info.get("model", "unknown")
    st.success(f"Connected: {model} ({provider})")
else:
    st.warning(
        "LLM not available. Add your `NVIDIA_API_KEY` to `.env` and restart:\n\n"
        "```\nNVIDIA_API_KEY=nvapi-...\n```\n\n"
        "`uvicorn api.main:app --reload`"
    )

st.divider()

# ------------------------------------------------------------------
# Chat settings
# ------------------------------------------------------------------
with st.expander("Settings"):
    temperature = st.slider("Temperature", 0.0, 2.0, 1.0, 0.1)
    max_tokens = st.slider("Max Tokens", 256, 8192, 8192, 256)
    enable_tools = st.toggle("Enable tool calling", value=True)

# ------------------------------------------------------------------
# Quick analysis buttons
# ------------------------------------------------------------------
st.subheader("Quick Analysis")
quick_col1, quick_col2, quick_col3 = st.columns(3)

with quick_col1:
    if st.button("Analyze Last Valuation", use_container_width=True):
        if st.session_state.last_valuation:
            result = api.llm_analyze(
                analysis_type="asset",
                context=st.session_state.last_valuation,
            )
            if result:
                st.session_state.setdefault("chat_messages", [])
                st.session_state.chat_messages.append({"role": "user", "content": "Analyze my last valuation result."})
                st.session_state.chat_messages.append({"role": "assistant", "content": result["text"]})
        else:
            st.info("Run a valuation first on the Asset Valuation page.")

with quick_col2:
    if st.button("Analyze Last Climate Risk", use_container_width=True):
        if st.session_state.last_climate_risk:
            result = api.llm_analyze(
                analysis_type="climate",
                context=st.session_state.last_climate_risk,
            )
            if result:
                st.session_state.setdefault("chat_messages", [])
                st.session_state.chat_messages.append({"role": "user", "content": "Analyze my last climate risk assessment."})
                st.session_state.chat_messages.append({"role": "assistant", "content": result["text"]})
        else:
            st.info("Run a climate risk assessment first.")

with quick_col3:
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_messages = []
        st.rerun()

st.divider()

# ------------------------------------------------------------------
# Chat interface
# ------------------------------------------------------------------
st.subheader("Chat")

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# Display chat history
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        # Show tool calls if present
        if msg.get("tool_calls"):
            with st.expander(f"Tools used ({len(msg['tool_calls'])})", expanded=False):
                for tc in msg["tool_calls"]:
                    st.markdown(f"**{tc['tool']}**")
                    st.json(tc.get("arguments", {}))
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about renewable energy, climate risk, valuations..."):
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build context from recent analyses
    context_parts = []
    if st.session_state.last_valuation:
        v = st.session_state.last_valuation
        context_parts.append(
            f"Recent valuation: NPV=${v['npv_usd']:,.0f}, IRR={v['irr']*100:.1f}%, "
            f"LCOE=${v['lcoe_per_mwh']:.2f}/MWh, asset={v['asset_id']}"
        )
    if st.session_state.last_climate_risk:
        r = st.session_state.last_climate_risk
        context_parts.append(
            f"Recent climate risk: score={r['risk_score']:.3f}, "
            f"solar_ghi_p50={r['solar_ghi_kwh_m2_year']['p50']:.0f}"
        )

    api_messages = []
    if context_parts:
        system_msg = (
            "You are an AI assistant specialized in renewable energy analysis. "
            "Here is context from the user's recent analyses:\n" + "\n".join(context_parts)
        )
        api_messages.append({"role": "system", "content": system_msg})

    recent = st.session_state.chat_messages[-10:]
    api_messages.extend(recent)

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = api.llm_chat(
                api_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                enable_tools=enable_tools,
            )

        if result:
            response = result["text"]
            tool_calls = result.get("tool_calls", [])

            # Show tool calls
            if tool_calls:
                with st.expander(f"Tools used ({len(tool_calls)})", expanded=True):
                    for tc in tool_calls:
                        st.markdown(f"**{tc['tool']}**")
                        st.json(tc.get("arguments", {}))
                        if "error" in tc.get("result", {}):
                            st.error(tc["result"]["error"])

            st.markdown(response)
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": response,
                "tool_calls": tool_calls,
            })
            st.caption(f"Model: {result.get('model', 'unknown')} | Tokens: {result.get('completion_tokens', '?')}")
        else:
            st.error("Failed to get a response. Is the LLM running?")
