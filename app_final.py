import streamlit as st
import numpy as np
import os
import base64
import fitz  # PyMuPDF for PDF processing
import speech_recognition as sr
from gtts import gTTS
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
import google.generativeai as genai
import google.api_core.exceptions  
import langdetect
import qrcode
from io import BytesIO
from PIL import Image
import socket
import geocoder
import re
import pythoncom
import pyttsx3
import base64
from gtts import gTTS
import io
import time
import pytesseract
import easyocr
import urllib.parse
import cv2
import pyperclip
import urllib.parse
import pandas as pd
from datetime import datetime, timedelta
from io import BytesIO
from PIL import Image
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
from email.message import EmailMessage
import streamlit.components.v1 as components
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from twilio.rest import Client
from fpdf import FPDF  # PDF generation
from duckduckgo_search import DDGS
from youtubesearchpython import VideosSearch
from deep_translator import GoogleTranslator
import streamlit as st
import random

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY_HERE")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Function to highlight text
def highlight_text(text):
    return f'<span style="background-color: #d4edda; padding: 2px 4px; border-radius: 4px;">{text}</span>'

# Function to bold text
def bold_text(text):
    return f'**{text}**'

# Function to extract sources from text
def extract_sources(text):
    source_pattern = r'\[(.*?)\]\((.*?)\)'
    matches = re.findall(source_pattern, text)
    sources = "\n".join([f"🔗 [{title}]({url})" for title, url in matches])
    cleaned_text = re.sub(source_pattern, '', text).strip()
    return cleaned_text, sources

# Function to fetch images from DuckDuckGo
def fetch_duckduckgo_images(query, max_results=1):
    with DDGS() as ddgs:
        results = [r['image'] for r in ddgs.images(query, max_results=max_results)]
    return results

def fetch_youtube_videos(query, max_results=1):
    video_results = []

    with DDGS() as ddgs:
        search_results = ddgs.text(f"{query} video site:youtube.com", max_results=20)
        for result in search_results:
            href = result.get('href', '')
            if 'youtube.com' in href and "/watch" in href:
                video_thumbnail = fetch_video_thumbnail(href)  # Fetch thumbnail for the video
                video_results.append({
                    "title": result.get('title', 'No Title'),
                    "link": href,
                    "thumbnail": video_thumbnail  # Add thumbnail to result
                })
            if len(video_results) >= max_results:
                break

    return video_results

def fetch_video_thumbnail(video_url):
    """
    This function fetches the thumbnail of the video from the provided YouTube URL.
    """
    if 'youtube.com' in video_url:
        video_id = video_url.split('v=')[-1].split('&')[0]
        return f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
    return ""  # Return empty if thumbnail cannot be found

# Remove this function completely
# def highlight_text(text):
#     return f'<span style="background-color: #d4edda; padding: 2px 4px; border-radius: 4px;">{text}</span>'

def get_gemini_response(question, lang="en"):
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        history = st.session_state.get("history", [])
        context = "\n".join([f"User: {entry['query']}\nBot: {entry['response']}" for entry in history[-5:]])

        full_prompt = f"""
        You are an AI assistant providing legal information.

        Conversation so far:
        {context}

        User's latest question:
        {question}

        Respond concisely. Include sources at the end in markdown format like [Source Title](https://example.com) for any external sources.
        """

        response = model.generate_content(full_prompt)
        answer_text = response.text.strip() if response else "Error fetching response."
        answer_text, sources = extract_sources(answer_text)

        # Bold formatting (optional)
        final_answer = re.sub(r'\[HIGHLIGHT\](.*?)\[/HIGHLIGHT\]', lambda m: bold_text(m.group(1)), answer_text)

        # Translate the answer if needed
        if lang != "en":
            translated = GoogleTranslator(source='auto', target=lang).translate(final_answer)
            question = GoogleTranslator(source='auto', target='en').translate(question)
        else:
            translated = final_answer

        # Display in Streamlit
        st.markdown(f"**You:** {question}")
        st.markdown(f"**Bot ({lang.upper()}):** {translated}")
        if sources:
            st.markdown(f"### Sources 📚\n{sources}")

        # Show relevant images
        image_urls = fetch_duckduckgo_images(question)
        if image_urls:
            st.markdown("### 🖼️ Relevant Images")
            for img_url in image_urls:
                st.image(img_url, caption="Relevant Image", use_container_width=True)

        # Show relevant YouTube videos with thumbnails
        video_results = fetch_youtube_videos(question)
        if video_results:
            st.markdown("### 🎥 Relevant YouTube Videos")
            for video in video_results:
                st.markdown(f"""
                <div style="margin-bottom: 15px;">
                    <a href="{video['link']}" target="_blank" style="text-decoration: none;">
                        <img src="{video['thumbnail']}" alt="Video Thumbnail" style="width: 100%; height: auto; border-radius: 8px;">
                        <p style="font-weight:bold; color: #1f77b4; margin-top: 10px;">🔗 {video['title']}</p>
                    </a>
                </div>
                """, unsafe_allow_html=True)

        # Update history
        st.session_state.history.append({"query": question, "response": translated})
        return translated

    except google.api_core.exceptions.ResourceExhausted:
        return "API quota exceeded. Please try again later."
    except Exception as e:
        return f"Error: {str(e)}"




def refine_video_query(question):
    model = genai.GenerativeModel("gemini-1.5-pro")
    prompt = f"""
    The user asked: "{question}"

    Provide a short and clear search query suitable for finding video results on platforms like YouTube, Vimeo, etc.
    Only return the search phrase—no explanations.
    """
    response = model.generate_content(prompt)
    return response.text.strip() if response else question



# Text-to-Speech (TTS) function
def text_to_speech():
    try:
        with open("index1.html", 'r', encoding='utf-8') as f:
            html_data = f.read()
        components.html(html_data, height=600, scrolling=True)
    except FileNotFoundError:
        st.error("index1.html not found. Make sure it's in the same directory as your app.")


# Call the function to embed the local HTML





# Function to split text into smaller chunks
# ✅ Place this function before summarize_pdf()
def redact_sensitive_info(text):
    """Uses Google Gemini AI to identify and redact sensitive info dynamically."""
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        prompt = f"Identify and replace any sensitive information in this text with '[REDACTED]'. Only modify sensitive details:\n\n{text}"
        response = model.generate_content(prompt)
        return response.text if response else text  # Return redacted text
    except Exception as e:
        return f"Error in redaction: {str(e)}"

# ✅ Now define summarize_pdf() after redact_sensitive_info()
def summarize_pdf(pdf_file, lang="en", redact=True):
    """Extracts and summarizes a PDF while optionally redacting sensitive info."""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        pdf_text = "\n".join(page.get_text() for page in doc)

    # Detect Language and Translate if Needed
    detected_lang = langdetect.detect(pdf_text)
    if detected_lang != lang:
        pdf_text = GoogleTranslator(source=detected_lang, target=lang).translate(pdf_text)

    # If redaction is enabled, redact sensitive info
    sanitized_text = redact_sensitive_info(pdf_text) if redact else pdf_text

    # Send text (redacted or original) to Gemini AI for summarization
    return get_gemini_response(f"Summarize this:\n\n{sanitized_text}", lang=lang)


                                                                               
def get_local_ip():
    hostname = socket.gethostname()
    return socket.gethostbyname(hostname)

def generate_qr_code():
    LOCAL_IP = get_local_ip()
    streamlit_url = f"http://{LOCAL_IP}:8501"
    qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=4)
    qr.add_data(streamlit_url)
    qr.make(fit=True)
    return qr.make_image(fill="black", back_color="white")


#LEGAL FIRM
def get_nearby_legal_firms_and_specialists(latitude, longitude, legal_need):
    if latitude and longitude:
        legal_firms_url = f"https://www.google.com/maps/search/legal+firms/@{latitude},{longitude},15z"
        specialist_url = f"https://www.google.com/maps/search/{legal_need}+lawyer/@{latitude},{longitude},15z"
        return legal_firms_url, specialist_url
    else:
        return None, None
def generate_contract(contract_type, details):
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        prompt = f"Generate a {contract_type} contract with the following details:\n{details}"
        response = model.generate_content(prompt)
        return response.text if response else "Error generating contract."
    except Exception as e:
        return f"Error: {str(e)}"

# Function to create a PDF contract
def create_pdf(contract_text, contract_type):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(190, 10, contract_text)

    pdf_filename = f"{contract_type}_contract.pdf"
    pdf.output(pdf_filename)
    return pdf_filename

# Define functions for different contract types
def generate_nda(details):
    return generate_contract("Non-Disclosure Agreement (NDA)", details)

def generate_lease(details):
    return generate_contract("Lease Agreement", details)

def generate_employment(details):
    return generate_contract("Employment Contract", details)
def save_as_pdf(text, filename="transcription.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(190, 10, text)

    pdf_path = f"./{filename}"
    pdf.output(pdf_path)
    return pdf_path


def text_to_speech():
    iframe_code = """
    <iframe src="index1.html" width="100%" height="600px" style="border:none;" title="Gemini Voice Assistant"></iframe>
    """
    return iframe_code


def legal_dictation():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Speak now... AI is listening.")
        recognizer.adjust_for_ambient_noise(source)

        try:
            audio = recognizer.listen(source, timeout=120)  # 10s timeout
            text = recognizer.recognize_google(audio)
            return text
        except sr.WaitTimeoutError:
            return "⏳ No speech detected, please try again."
        except sr.UnknownValueError:
            return "❌ Could not understand speech."
        except sr.RequestError:
            return "⚠️ Error connecting to speech recognition service."

def efiling_assistant():
    st.subheader("🧾 Dynamic eFiling Assistant")

    st.markdown("Please provide the details required to draft your legal complaint:")

    user_fields = st.number_input("How many input fields do you want?", min_value=1, max_value=15, step=1)
    inputs = {}

    for i in range(user_fields):
        label = st.text_input(f"Label for Field {i + 1}")
        value = st.text_area(f"Value for {label}", key=f"value_{i}")
        if label:
            inputs[label] = value

    if st.button("📄 Generate Complaint Draft"):
        st.markdown("### 📝 eFiling Complaint Draft")

        # Format custom fields as a bullet list
        formatted_fields = "\n".join([f"**{label}**: {value}" for label, value in inputs.items()])

        # Dynamic complaint template
        complaint_template = f"""
To,  
The Officer-in-Charge / Concerned Authority,  
[Your Local Jurisdiction Office],  

Date: [Insert Date]

Subject: **Legal Complaint Submission**

Respected Sir/Madam,

I am writing to formally submit a legal complaint with the following details:

{formatted_fields}

I kindly request you to take appropriate legal action in this regard. I am available to provide any additional information or clarification that may be required.

Thank you for your attention.

Sincerely,  
[Your Name]  
[Contact Information]
        """

        st.code(complaint_template.strip(), language="markdown")

        st.success("✅ Your legal complaint draft is ready!")

        # Optional: eFiling portal link
        st.markdown("### 🔗 [Proceed to eFiling Portal](https://efiling.ecourts.gov.in/)")







def preprocess_image(image_np, target_size=(512, 512)):
    """Resize the image to reduce memory usage before OCR processing."""
    resized_image = cv2.resize(image_np, target_size, interpolation=cv2.INTER_AREA)
    return resized_image

def auto_form_fill_from_id():
    st.subheader("🪪 Auto-Form Fill from Uploaded ID")

    uploaded_id = st.file_uploader("Upload your ID image (any type)", type=["jpg", "jpeg", "png"])

    if uploaded_id is not None:
        image = Image.open(uploaded_id).convert("RGB")
        image_np = np.array(image)
        image_np = preprocess_image(image_np)

        with st.spinner("🔍 Reading text..."):
            reader = easyocr.Reader(['en'])
            result = reader.readtext(image_np, detail=0)
            extracted_lines = [line.strip() for line in result if line.strip()]
            extracted_text = "\n".join(extracted_lines)

        # 🔎 Document Definitions with Keywords + Smart Pattern Rules
        doc_definitions = {
            "Aadhaar Card": {
                "keywords": ["aadhaar", "uidai", "government of india", "unique identification", "aadhaar no", "vid"],
                "patterns": [r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'],
                "link": "https://uidai.gov.in/en/my-aadhaar/get-aadhaar"
            },
            "PAN Card": {
                "keywords": ["income tax", "pan card", "permanent account number", "income tax department"],
                "patterns": [r'\b[A-Z]{5}[0-9]{4}[A-Z]\b'],
                "link": "https://www.incometax.gov.in/iec/foportal/"
            },
            "Driving License": {
                "keywords": ["driving licence", "transport department", "govt of india"],
                "patterns": [r'\b[A-Z]{2}[ -]?\d{2}[ -]?\d{4,7}\b'],
                "link": "https://parivahan.gov.in/parivahan/en/content/driving-licence"
            },
            "Voter ID": {
                "keywords": ["voter", "election commission", "epic no", "elector"],
                "patterns": [r'\b[A-Z]{3}[0-9]{7}\b'],
                "link": "https://voters.eci.gov.in/"
            },
            "Passport": {
                "keywords": ["passport", "ministry of external affairs", "passport seva"],
                "patterns": [r'\b[A-Z][0-9]{7}\b'],
                "link": "https://www.passportindia.gov.in/"
            }
        }

        # 🔍 Detect Document Type
        detected_doc_type = None
        portal_link = None

        for doc_type, details in doc_definitions.items():
            keyword_hit = any(keyword.lower() in extracted_text.lower() for keyword in details["keywords"])
            pattern_hit = any(re.search(pattern, extracted_text) for pattern in details["patterns"])

            if keyword_hit or pattern_hit:
                detected_doc_type = doc_type
                portal_link = details["link"]
                break

        # ✅ Display Structured Text
        if extracted_lines:
            st.success("✅ Text successfully extracted from the document!")

            if detected_doc_type:
                st.info(f"**Detected Document Type:** {detected_doc_type}")
                st.markdown(f"🔗 [Visit {detected_doc_type} Portal]({portal_link})")

            st.markdown("### 📄 Structured Extracted Data:")
            structured_data = {}

            for idx, line in enumerate(extracted_lines, 1):
                if ":" in line:
                    key, value = line.split(":", 1)
                    structured_data[key.strip()] = value.strip()
                else:
                    structured_data[f"Field {idx}"] = line

            for key, value in structured_data.items():
                st.write(f"**{key}:** {value}")
        else:
            st.warning("⚠️ No text could be extracted. Try a clearer image.")
def lawyer_video_call_and_payment():
    st.header("⚖️ Live Consultation & UPI Payment")

    lawyers = {
        "Adarsh Kumar": {"upi": "shreenidhibalaji2004@okhdfcbank", "room": "ConsultWith_Adarsh"},
        "Nisha Verma": {"upi": "nishav@oksbi", "room": "ConsultWith_Nisha"},
        "Ravi Singh": {"upi": "ravi.singh@ybl", "room": "ConsultWith_Ravi"}
    }

    selected = st.selectbox("Choose a lawyer:", list(lawyers.keys()))
    user_name = st.text_input("Enter your name (for record):")

    info = lawyers[selected]
    upi_url = f"upi://pay?pa={info['upi']}&pn={selected.replace(' ', '%20')}&am=500&cu=INR"

    qr = qrcode.make(upi_url)
    buffer = BytesIO()
    qr.save(buffer, format="PNG")
    buffer.seek(0)

    st.subheader("📹 Video Call")
    st.components.v1.html(f"""
        <iframe src="https://meet.jit.si/{info['room']}" 
        allow="camera; microphone" width="100%" height="600" frameborder="0"></iframe>
    """, height=600)

    st.subheader("💳 UPI Payment (₹500)")
    st.image(buffer, caption="Scan to pay via any UPI app")
    st.markdown(f"[👉 Click to Pay via GPay/PhonePe/Paytm]({upi_url})", unsafe_allow_html=True)

    paid = st.checkbox("✅ I have completed the payment")

    if paid and user_name:
        payment_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user": user_name,
            "lawyer": selected,
            "upi_id": info["upi"],
            "status": "Paid"
        }

        df = pd.DataFrame([payment_data])
        file_exists = os.path.exists("payments_log.csv")
        df.to_csv("payments_log.csv", mode='a', index=False, header=not file_exists)

        st.success("✅ Payment logged successfully!")


def appointment_booking():
    st.header("📅 Smart Appointment Booking")

    lawyers = {
        "Adarsh Kumar": ["10:00 AM", "11:30 AM", "2:00 PM"],
        "Nisha Verma": ["9:30 AM", "1:00 PM", "4:00 PM"],
        "Ravi Singh": ["11:00 AM", "3:00 PM", "5:30 PM"]
    }

    user_name = st.text_input("Enter your name:")
    user_email = st.text_input("Enter your email:")

    selected_lawyer = st.selectbox("Choose a lawyer:", list(lawyers.keys()))
    selected_date = st.date_input("Select a date:")
    selected_time = st.selectbox("Select a time slot:", lawyers[selected_lawyer])

    if st.button("📩 Confirm Appointment"):
        if user_name and user_email and selected_lawyer and selected_date and selected_time:
            appointment = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user": user_name,
                "email": user_email,
                "lawyer": selected_lawyer,
                "date": selected_date.strftime("%Y-%m-%d"),
                "time": selected_time,
                "status": "Booked"
            }

            df = pd.DataFrame([appointment])
            file_exists = os.path.exists("appointments_log.csv")
            df.to_csv("appointments_log.csv", mode='a', index=False, header=not file_exists)

            st.success(f"✅ Appointment booked with {selected_lawyer} on {selected_date} at {selected_time}.")

            # Email notification to lawyer/admin
            your_email = "youremail@gmail.com"
            subject = f"📢 New Appointment with {selected_lawyer}"
            body = f"""
New Appointment Booked:

👤 Name: {user_name}
📧 Email: {user_email}
⚖️ Lawyer: {selected_lawyer}
📅 Date: {selected_date.strftime("%Y-%m-%d")}
⏰ Time: {selected_time}
"""
            if send_email_notification(your_email, subject, body):
                st.info("📧 Notification email sent!")
            else:
                st.warning("⚠️ Failed to send email notification.")
        else:
            st.warning("⚠️ Please fill out all the fields.")


def appointment_booking():
    st.header("📆 Schedule Your Appointment")

    # Static values
    lawyer_name = "Shree"
    calendly_link = "https://calendly.com/2022cs0344-svce"

    st.markdown(f"### 👤 Schedule an appointment with ")

    # Styled HTML button link that opens in new tab
    st.markdown(f"""
        <a href="{calendly_link}" target="_blank">
            <button style='padding:10px 20px; font-size:16px; background-color:#4CAF50; color:white; border:none; border-radius:8px; cursor:pointer;'>
                📅 Book Now
            </button>
        </a>
    """, unsafe_allow_html=True)

    with st.expander("🔗 What Happens After Booking?"):
        st.markdown("""
        - ✅ **Zoom Link** will be sent to your email once you book the appointment.
        - 🔁 **Reminder emails** will be sent 24 hours and 1 hour before the session.
        - 💳 **If payment is required**, you'll be redirected before confirmation.

        ---
        🔧 *Make sure these are enabled in your Calendly settings:*
        - Zoom Integration via Calendly > Integrations
        - Email reminders under Event Notifications
        - Payment via Calendly + Stripe/PayPal
        """)


def dictation_and_transcription():
    st.markdown("""
        <iframe src="http://127.0.0.1:5500/index.html"
                width="100%" 
                height="500" 
                frameborder="0" 
                allow="microphone"
                sandbox="allow-scripts allow-same-origin allow-forms allow-popups allow-modals allow-presentation allow-top-navigation allow-popups-to-escape-sandbox allow-downloads allow-pointer-lock allow-downloads-without-user-activation allow-storage-access-by-user-activation allow-scripts allow-same-origin allow-presentation allow-forms allow-top-navigation-by-user-activation allow-modals">
        </iframe>
    """, unsafe_allow_html=True)

def show_case_status_tab():
    st.markdown(
        """
        <div style="text-align:center; padding: 20px;">
            <h2 style="color:#2c3e50;">🔎 Check Court Case Status</h2>
            <p style="font-size:18px;">Enter your <strong>16-digit CNR number</strong> on the official eCourts page to get your case status.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="text-align:center; margin-bottom: 30px;">
            <p style="font-size:16px; color:#555;">Example CNR Number:</p>
            <div style="display: inline-block; background-color: #f4f4f4; padding: 10px 20px; border-radius: 10px; font-size: 18px; font-weight: bold; color:#000;">
                MHAU019999992015
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="text-align:center;">
            <a href="https://services.ecourts.gov.in/ecourtindia_v6/" target="_blank">
                <button style="
                    background: linear-gradient(to right, #4CAF50, #2E7D32);
                    color: white;
                    padding: 12px 28px;
                    font-size: 18px;
                    border: none;
                    border-radius: 8px;
                    cursor: pointer;
                    transition: background 0.3s ease;">
                    🧾 Go to CNR Case Status Page
                </button>
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="text-align:center; padding-top: 20px;">
            <small style="color: gray;">
                On the next page, select the <strong>CNR Number</strong> tab, enter your code and complete the captcha to view your case status.
            </small>
        </div>
        """, 
        unsafe_allow_html=True
    )

def voice_assistant(html_path="index1.html", height=600, width=800):
    """Embeds a local HTML page in Streamlit using an iframe."""
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    components.html(html_content, height=height, width=width, scrolling=True)



def show_emergency_helpline():
    st.markdown("""
        <style>
            .helpline-container {
                background: linear-gradient(145deg, #f7f7f7, #e0e0e0);
                border-radius: 16px;
                padding: 20px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                margin-top: 20px;
            }
            .helpline-title {
                font-size: 28px;
                font-weight: 700;
                color: #d9534f;
                text-align: center;
                margin-bottom: 20px;
            }
            .helpline-button {
                display: block;
                width: 100%;
                max-width: 400px;
                margin: 10px auto;
                padding: 15px 30px;
                background: #d9534f;
                color: white;
                font-size: 18px;
                font-weight: bold;
                border: none;
                border-radius: 12px;
                transition: all 0.3s ease;
                text-align: center;
                text-decoration: none;
                box-shadow: 0 4px 14px rgba(217,83,79,0.4);
            }
            .helpline-button:hover {
                background: #c9302c;
                box-shadow: 0 6px 20px rgba(217,83,79,0.6);
                transform: scale(1.03);
            }
        </style>
        <div class="helpline-container">
            <div class="helpline-title">🚨 Emergency Helpline Numbers</div>
    """, unsafe_allow_html=True)
    
    emergency_contacts = {
        "🚓 Police": {"number": "100", "url": "https://police.gov.in"},
        "🚑 Ambulance": {"number": "102", "url": "https://www.bing.com/search?q=ambulance+portal&cvid=0046d545ed484565925395556f01b741&gs_lcrp=EgRlZGdlKgYIABBFGDkyBggAEEUYOTIGCAEQABhAMgYIAhAAGEAyBggDEAAYQDIGCAQQABhAMgYIBRAAGEAyBggGEAAYQDIGCAcQABhAMgYICBAAGEDSAQg0ODg3ajBqNKgCCLACAQ&FORM=ANAB01&PC=U531"},
        "🔥 Fire": {"number": "101", "url": "https://ndma.gov.in"},
        "🌀 Disaster Management": {"number": "108", "url": "https://ndma.gov.in"},
        "👩‍🦰 Women Helpline": {"number": "1091", "url": "https://ncw.nic.in"},
        "👶 Child Helpline": {"number": "1098", "url": "https://www.childlineindia.org"},
        "🕵️‍♂️ Cyber Crime": {"number": "1930", "url": "https://cybercrime.gov.in"},
        "👴 Senior Citizens": {"number": "14567", "url": "https://sje.gov.in"},
        "⚖️ Legal Aid": {"number": "15100", "url": "https://nalsa.gov.in"},
        "🦠 AIDS Helpline": {"number": "1097", "url": "https://www.naco.gov.in"},
        "🚘 Road Accident": {"number": "1073", "url": "https://morth.nic.in"},
        "🚆 Railways": {"number": "139", "url": "https://enquiry.indianrail.gov.in"},
        "🧠 Mental Health": {"number": "08046110007", "url": "https://telemanas.mohfw.gov.in"}
    }

    for service, details in emergency_contacts.items():
        st.markdown(f"""
            <a class="helpline-button" href="tel:{details['number']}" target="_blank">
                {service} (Call: {details['number']})
            </a>
            <a class="helpline-button" href="{details['url']}" target="_blank">
                🌐 Visit {service.strip('🚓🚑🔥🌀👩‍🦰👶🕵️‍♂️👴⚖️🦠🚘🚆🧠')} Portal
            </a>
        """, unsafe_allow_html=True)

    st.markdown("""</div>""", unsafe_allow_html=True)
    st.info("📱 Click-to-call works on smartphones or VoIP-enabled systems. Portal links open in a new tab.")



def start_random_legal_quiz():
    question_bank = [
    {"q": "What is the fundamental duty of every Indian citizen?", "a": "To respect the Constitution"},
    {"q": "Which Article guarantees Right to Education?", "a": "Article 21A"},
    {"q": "What is PIL in legal terms?", "a": "Public Interest Litigation"},
    {"q": "Who appoints the Chief Justice of India?", "a": "President"},
    {"q": "What does IPC stand for?", "a": "Indian Penal Code"},
    {"q": "What is the minimum age for marriage for girls in India?", "a": "18"},
    {"q": "What court is highest in India?", "a": "Supreme Court"},
    {"q": "RTI stands for?", "a": "Right to Information"},
    {"q": "Which law prohibits child labour?", "a": "Child Labour Act"},
    {"q": "Is dowry legal in India?", "a": "No"}
]
    
    st.markdown("## 🧠  Legal Quiz ")
    st.info("Answer as many as you can within the time! You get **15 seconds per question**.")

    total_questions = 5
    quiz_questions = random.sample(question_bank, total_questions)

    if 'quiz_index' not in st.session_state:
        st.session_state.quiz_index = 0
        st.session_state.score = 0
        st.session_state.timer_start = time.time()

    current_index = st.session_state.quiz_index
    if current_index >= total_questions:
        st.balloons()
        st.success(f"🏁 Quiz Over! Your Score: {st.session_state.score} / {total_questions}")
        if st.session_state.score == total_questions:
            st.markdown("🏅 You earned the **Legal Mastermind Badge!**")
        if st.button("🔄 Play Again"):
            st.session_state.quiz_index = 0
            st.session_state.score = 0
            st.session_state.timer_start = time.time()
        return

    question = quiz_questions[current_index]
    st.markdown(f"### ❓ Q{current_index+1}: {question['q']}")

    # Timer display
    seconds_left = 15 - int(time.time() - st.session_state.timer_start)
    if seconds_left > 0:
        st.warning(f"⏳ Time left: {seconds_left} seconds")
        user_ans = st.text_input("Your answer:", key=f"ans_{current_index}")

        if st.button("Submit"):
            if user_ans.strip().lower() == question["a"].lower():
                st.success("✅ Correct!")
                st.session_state.score += 1
            else:
                st.error(f"❌ Incorrect! Correct answer: {question['a']}")

            st.session_state.quiz_index += 1
            st.session_state.timer_start = time.time()
            st.rerun()  # Use st.rerun() instead of st.experimental_rerun()
    else:
        st.error("⏰ Time's up!")
        st.session_state.quiz_index += 1
        st.session_state.timer_start = time.time()
        st.rerun()  # Use st.rerun() instead of st.experimental_rerun()

def show_legal_aid_map():
    st.markdown("## 🗺️ State-wise Justice Delivery: Legal Aid (Large States)")
    st.markdown("""
        The interactive map below is available on the **India Justice Report** website.
        It visualizes how well large states are performing in providing **legal aid**, 
        aligning with **SDG 16: Peace, Justice & Strong Institutions**.
    """)

    st.markdown(
        """
        <a href="https://indiajusticereport.org/rankings/ijr-4/legal_aid/large-states/map" target="_blank">
            <button style='padding:10px 20px; font-size:16px; background-color:#4CAF50; color:white; border:none; border-radius:5px; cursor:pointer;'>
                🌐 View Legal Aid Map
            </button>
        </a>
        """,
        unsafe_allow_html=True
    )








st.set_page_config(page_title="AI Assistant", layout="wide")

# Initialize session state for chat history and selected chat
if "history" not in st.session_state:
    st.session_state.history = []
if "selected_chat" not in st.session_state:
    st.session_state.selected_chat = None
if "new_chat" not in st.session_state:
    st.session_state.new_chat = False

# Sidebar: Chat History Management
st.sidebar.title("📜 Chat History")

# "New Chat" Button
if st.sidebar.button("➕ New Chat"):
    st.session_state.selected_chat = None  # Deselect previous chat
    st.session_state.new_chat = True  # Enable new chat mode
    st.rerun()

# Display stored chat history in the sidebar
if st.session_state.history:
    for i, item in enumerate(st.session_state.history):
        if st.sidebar.button(f"🔹 {item['query']}", key=f"chat_{i}"):
            st.session_state.selected_chat = item  # Store selected chat in session state
            st.session_state.new_chat = False  # Disable new chat mode
            st.rerun()

# Clear History Button
if st.sidebar.button("🗑️ Clear History"):
    st.session_state.history = []
    st.session_state.selected_chat = None
    st.session_state.new_chat = True  # Enable new chat mode
    st.rerun()

# Sidebar Features
st.sidebar.title("🔍 Features")
selected_tab = st.sidebar.radio(
    "Choose a feature:",
    ["💬 AI Chat", "📄 Summarize PDF", 
     "📱 QR Code Generator", "📍Access Nearby", "Contract Drafting", "🖊️ Legal Dictation","📥 eFiling Assistant","🪪 Auto Form Fill","🧑‍⚖️ Live Video Call & Pay","Appointment Booking","dictation_and_transcription","Case Status","voice_assistant","🚨 Emergency Helpline","🧠 Random Legal Quiz","🗺️ State-wise Legal Aid Map",
]
)

# Main Workspace Title
st.title("🤖 Legal AI Assistant")
st.divider()

# Chat Input Section (for new chats)
if st.session_state.new_chat:
    user_input = st.text_input("💬 Type your message:", "")
    if st.button("Send"):
        if user_input:
            response = f"🔍 AI Response to: {user_input}"  # Replace with actual AI function call
            chat_entry = {"query": user_input, "response": response}
            st.session_state.history.append(chat_entry)
            st.session_state.selected_chat = chat_entry
            st.session_state.new_chat = False  # Disable new chat mode
            st.rerun()

# Display selected chat on the right
if st.session_state.selected_chat:
    query = st.session_state.selected_chat['query']
    response = st.session_state.selected_chat['response']

    chat_display = f"""
    <div class="chat-display">
        <div class="chat-box query-box">🔍 {query}</div>
        <div class="chat-box response-box">💬 {response}</div>
    </div>
    """

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.markdown(chat_display, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if selected_tab == "Appointment Booking":
    # Only show this section when user clicks 'Appointment Booking'
    st.sidebar.markdown("### 📂 Go to")
    sub_option = st.sidebar.radio("Select:", [
        "Book Appointment (Calendly)", 
        "Consult & Pay"
    ])

    if selected_tab == "Book Appointment (Calendly)":
        appointment_booking()
    elif selected_tab == "Consult & Pay":
        lawyer_video_call_and_payment()

if selected_tab == "💬 AI Chat":
    st.subheader("💬 AI Chat Assistant")

    user_query = st.chat_input("Ask me anything...")

    if user_query:
        target_lang = st.selectbox("Select language:", ["en", "hi", "fr", "es", "de", "ta", "te"], key="chat_lang")

        # Placeholder for bot response animation
        placeholder = st.empty()
        placeholder.markdown(
            """
            <div style="text-align:center; font-size:24px;">
                😊 <span class="dot">.</span><span class="dot">.</span><span class="dot">.</span>
            </div>
            <style>
                @keyframes bounce {
                    0%, 100% { transform: translateY(0); }
                    50% { transform: translateY(-5px); }
                }
                .dot {
                    display: inline-block;
                    font-size: 30px;
                    animation: bounce 1.5s infinite ease-in-out alternate;
                }
                .dot:nth-child(2) { animation-delay: 0.2s; }
                .dot:nth-child(3) { animation-delay: 0.4s; }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Fetch response from Gemini AI
        answer = get_gemini_response(user_query, target_lang)

        # Replace animation with actual response
        placeholder.empty()
        st.session_state.history.append({"query": user_query, "response": answer, "feature": "Chat"})

        # Styled response display
        st.markdown(f"**🧑‍💼 You:** {user_query}")
        st.markdown(f"**🤖 Bot:** {answer}")

### 🎙️ VOICE ASSISTANT ###
elif selected_tab == "🎙️ Voice Assistant":
    text_to_speech()


    
elif selected_tab == "📄 Summarize PDF":
    st.subheader("📄 Summarize PDF Document")

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        # Call the summarize_pdf function
        summary = summarize_pdf(uploaded_file)

        # Display the summary
        st.markdown("### 📘 Summary")
        st.write(summary)

        # Ask a question based on the summary
        st.markdown("### ❓ Ask a Question Based on the Summary")
        user_question = st.text_input("What do you want to know from the above summary?")

        if user_question:
            with st.spinner("🤖 Thinking..."):
                prompt = f"Based on the following summary:\n\n{summary}\n\nAnswer the question: {user_question}"
                answer = get_gemini_response(prompt)
                st.markdown("### 💬 Answer")
                st.write(answer)


### 📱 QR CODE GENERATOR ###
elif selected_tab == "📱 QR Code Generator":
    st.subheader("📱 Scan QR Code to Open App")
    
    if st.button("📷 Generate QR Code"):
        qr_image = generate_qr_code()
        img_bytes = BytesIO()
        qr_image.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        st.image(img_bytes, caption="Scan to open app", use_container_width=True)


### 📍 FIND NEARBY LEGAL FIRMS ###
elif selected_tab == "📍Access Nearby":  # Ensure exact match in sidebar
    st.subheader("📍 Find Nearby Legal Firms")

    # Store location persistently inside this section only
    if "user_location" not in st.session_state:
        st.session_state.user_location = None

    # Button to detect location
    if st.button("🔎 Detect My Location"):
        st.write("🔍 Detecting location...")  # Debugging message
        g = geocoder.ip('me')  # Fetch user location

        if g.latlng:
            st.session_state.user_location = g.latlng
            st.success(f"📍 Location Detected: {st.session_state.user_location[0]}, {st.session_state.user_location[1]}")
        else:
            st.error("❌ Could not retrieve location. Please check your internet connection.")
            st.write("Debug: Location retrieval failed, g.latlng returned None.")  # Extra debug output

    # Ensure location is retrieved before asking for legal needs
    if st.session_state.user_location:
        legal_need = st.text_input("Enter your legal need (e.g., 'divorce', 'criminal', 'corporate'):")
        
        if st.button("🔍 Find Nearby Legal Firms") and legal_need:
            latitude, longitude = st.session_state.user_location
            legal_firms_url, specialist_url = get_nearby_legal_firms_and_specialists(latitude, longitude, legal_need)

            st.markdown(f"🌍 [Nearby Legal Firms]({legal_firms_url})")
            st.markdown(f"🔎 [Specialist Lawyers for {legal_need}]({specialist_url})")
        elif not legal_need:
            st.warning("⚠️ Please enter a specific legal need.")
elif selected_tab == "Contract Drafting":
    st.subheader("📝 Generate a Legal Contract")

    contract_type = st.selectbox("Select Contract Type", ["NDA", "Lease", "Employment"])
    details = st.text_area("Enter contract details (e.g., parties involved, terms, duration)")

    if st.button("📝 Generate Contract"):
        if details:
            if contract_type == "NDA":
                contract_text = generate_nda(details)
            elif contract_type == "Lease":
                contract_text = generate_lease(details)
            elif contract_type == "Employment":
                contract_text = generate_employment(details)
            else:
                contract_text = "Invalid contract type selected."

            st.subheader("📜 Generated Contract")
            st.text_area("Contract", contract_text, height=300)

            # Save as PDF
            pdf_file = create_pdf(contract_text, contract_type)
            with open(pdf_file, "rb") as file:
                st.download_button("📥 Download Contract", file, file_name=pdf_file, mime="application/pdf")
        else:
            st.warning("⚠️ Please enter contract details.")
elif selected_tab == "🖊️ Legal Dictation":
    st.subheader("📝 Legal Dictation – Speak, and AI Writes It Down")

    if st.button("🎤 Start Dictation"):
        transcribed_text = legal_dictation()
        st.text_area("📄 Transcribed Text:", transcribed_text, height=200)

        if transcribed_text and not transcribed_text.startswith(("⏳", "❌", "⚠️")):  # Avoid errors for invalid text
            # ✅ Save as PDF
            pdf_file = save_as_pdf(transcribed_text)
            with open(pdf_file, "rb") as file:
                st.download_button("📥 Download as PDF", file, file_name="Legal_Dictation.pdf", mime="application/pdf")

elif selected_tab == "📥 eFiling Assistant":
    efiling_assistant()

elif selected_tab == "🪪 Auto Form Fill":
    auto_form_fill_from_id()



elif selected_tab == "🧑‍⚖️ Live Video Call & Pay":
    lawyer_video_call_and_payment()





if selected_tab == "Appointment Booking":
    appointment_booking()

elif selected_tab == "Consult & Pay":
    lawyer_video_call_and_payment()  # Make sure this function is defined elsewhere

elif selected_tab == "dictation_and_transcription":
    dictation_and_transcription()


elif selected_tab == "Case Status":
    show_case_status_tab()

elif selected_tab == "voice_assistant":
    st.subheader("your voice assistant")
    voice_assistant("index1.html", height=700, width=1000)
elif selected_tab == "🚨 Emergency Helpline":
    show_emergency_helpline()
elif selected_tab == "🧠 Random Legal Quiz":
    start_random_legal_quiz()
elif selected_tab == "🗺️ State-wise Legal Aid Map":
    show_legal_aid_map()
