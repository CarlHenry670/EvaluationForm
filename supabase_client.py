# supabase_client.py
import os
from supabase import create_client, Client

def get_client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        import streamlit as st
        url = url or st.secrets["SUPABASE_URL"]
        key = key or st.secrets["SUPABASE_SERVICE_ROLE_KEY"]
    return create_client(url, key)
