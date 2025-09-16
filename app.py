import gradio as gr
import requests
import pandas as pd
import smtplib
from email.mime.text import MIMEText
import os

# Keys
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
MAILJET_API_KEY = os.environ.get("MAILJET_API_KEY")
MAILJET_SECRET_KEY = os.environ.get("MAILJET_SECRET_KEY")

def send_email(subject, body, to_email,
               smtp_server="in-v3.mailjet.com", smtp_port=587):
    """Send summary via email using Mailjet SMTP"""
    try:
        msg = MIMEText(body)
        msg["From"] = "ssabn001@gmail.com"   # Replace with verified domain if possible
        msg["To"] = to_email
        msg["Subject"] = subject

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(MAILJET_API_KEY, MAILJET_SECRET_KEY)
            server.sendmail("ssabn001@gmail.com", to_email, msg.as_string())

        return f"✅ Summary sent to {to_email}"
    except Exception as e:
        return f"❌ Email failed: {e}"

def search_flights(origin, destination, outbound_date, return_date, user_email):
    # --- Step 1: Fetch flights from SerpAPI ---
    url = "https://serpapi.com/search"
    params = {
        "engine": "google_flights",
        "departure_id": origin,
        "arrival_id": destination,
        "outbound_date": outbound_date,
        "return_date": return_date,
        "currency": "USD",
        "api_key": SERPAPI_KEY,
    }

    resp = requests.get(url, params=params, timeout=30, verify=False)
    if resp.status_code != 200:
        return f" SerpAPI error {resp.status_code}: {resp.text}", "", ""

    data = resp.json()
    flights = data.get("best_flights", []) + data.get("other_flights", [])

    if not flights:
        return "⚠️ No flights found. Try different dates or airports.", "", ""

    # --- Step 2: Convert JSON → DataFrame ---
    df = pd.json_normalize(
        flights,
        record_path=["flights"],  # each leg inside the flight option
        meta=["price"],           # keep top-level price
        errors="ignore"
    )

    if df.empty:
        return "⚠️ Could not parse flight data.", "", ""

    # --- Step 3: Summarize flights ---
    flight_summary = df.to_string(
        columns=["airline", "duration", "departure_airport.id", "arrival_airport.id", "price"],
        index=False
    )

    system_prompt = (
        "You are a travel assistant. "
        "Choose the best flight based on price, duration, and number of stops.\n\n"
        f"{flight_summary}"
    )

    user_question = "Which flight is the best option and why?"

    # --- Step 4: Call Mistral REST API ---
    mistral_url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "mistral-small-latest",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ]
    }

    mistral_resp = requests.post(mistral_url, headers=headers, json=payload, verify=False)
    if mistral_resp.status_code == 200:
        result = mistral_resp.json()
        recommendation = result["choices"][0]["message"]["content"]
    else:
        recommendation = f" Mistral error {mistral_resp.status_code}: {mistral_resp.text}"

    # --- Step 5: Email summary via Mailjet ---
    email_status = send_email(
        subject="Your Flight Recommendation",
        body=recommendation,
        to_email=user_email
    )

    return flight_summary, recommendation, email_status


# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("## **✈️ Best Flight Finder and Travel Planner with Mistral AI**", elem_classes="")
    with gr.Row():
        origin = gr.Textbox(label="Origin Airport Code (e.g., IAD)")
        destination = gr.Textbox(label="Destination Airport Code (e.g., SFO)")
    with gr.Row():
        outbound_date = gr.Textbox(label="Outbound Date (YYYY-MM-DD)")
        return_date = gr.Textbox(label="Return Date (YYYY-MM-DD)")

    user_email = gr.Textbox(label="📧 Recipient Email (where summary will be sent)")

    search_btn = gr.Button("🔍 Search Flights & Email Summary")
    flights_output = gr.Textbox(label="Flights Found", lines=10)
    recommendation_output = gr.Textbox(label="🤖 AI Recommendation", lines=6)
    email_status_output = gr.Textbox(label="📩 Email Status", lines=2)

    search_btn.click(
        fn=search_flights,
        inputs=[origin, destination, outbound_date, return_date, user_email],
        outputs=[flights_output, recommendation_output, email_status_output]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    demo.launch(server_name="0.0.0.0", server_port=port)
