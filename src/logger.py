import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import streamlit as st

def log_to_google_sheets(user_input, bot_response, sources, session_id):
    try:
        creds_dict = st.secrets["gcp_service_account"]

        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]

        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)

        sheet = client.open("mumma-bot-logs").sheet1

        sheet.append_row([
            str(datetime.now()),
            session_id,
            user_input,
            bot_response,
            ", ".join(sources)
        ])

    except Exception as e:
        print("Logging failed:", e)