import streamlit as st
import pytesseract
import cv2
import re
import numpy as np
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.set_page_config(
    page_title="TravelDoc AI",
    page_icon="✈️",
    layout="wide"
)

st.title("TravelDoc AI")
st.subheader("OCR-Based Travel Ticket Information Extractor")

st.write(
    "Upload a train, bus, or flight ticket image to extract important travel information."
)

uploaded_file = st.file_uploader(
    "Upload your ticket image",
    type=["jpg", "jpeg", "png"]
)

def preprocess_image(image):
    image = np.array(image)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    gray = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    processed = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    return processed


def extract_information(text):

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    information = {
        "Passenger Name": "Not detected",
        "Booking / PNR Number": "Not detected",
        "Travel Date": "Not detected",
        "Departure Time": "Not detected",
        "Arrival Time": "Not detected",
        "Source": "Not detected",
        "Destination": "Not detected",
        "Seat / Coach": "Not detected"
    }

    for line in lines:
        lower = line.lower()

        if any(word in lower for word in ["passenger", "traveller", "traveler", "name"]):
            value = re.sub(
                r"(?i)(passenger|traveller|traveler|name)\s*[:\-]?\s*",
                "",
                line
            )

            if value:
                information["Passenger Name"] = value

        elif any(word in lower for word in ["pnr", "booking", "reservation"]):
            value = re.sub(
                r"(?i)(pnr|booking|reservation)(\s*(number|no|id))?\s*[:\-]?\s*",
                "",
                line
            )

            if value:
                information["Booking / PNR Number"] = value

        elif any(word in lower for word in ["date", "journey date", "travel date"]):
            match = re.search(
                r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
                line
            )

            if match:
                information["Travel Date"] = match.group()

        elif any(word in lower for word in ["departure", "depart"]):
            match = re.search(
                r"\b\d{1,2}[:.]\d{2}\s*(?:AM|PM)?\b",
                line,
                re.IGNORECASE
            )

            if match:
                information["Departure Time"] = match.group()

        elif any(word in lower for word in ["arrival", "arrive"]):
            match = re.search(
                r"\b\d{1,2}[:.]\d{2}\s*(?:AM|PM)?\b",
                line,
                re.IGNORECASE
            )

            if match:
                information["Arrival Time"] = match.group()

        elif any(word in lower for word in ["seat", "coach", "berth"]):
            value = re.sub(
                r"(?i)(seat|coach|berth)\s*[:\-]?\s*",
                "",
                line
            )

            if value:
                information["Seat / Coach"] = value

    return information


if uploaded_file:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Ticket")
        st.image(image, use_container_width=True)

    with st.spinner("Processing ticket..."):

        processed_image = preprocess_image(image)

        extracted_text = pytesseract.image_to_string(
            processed_image
        )

        information = extract_information(extracted_text)

    with col2:

        st.subheader("Extracted Travel Information")

        for key, value in information.items():
            st.write(f"**{key}:** {value}")

    st.subheader("OCR Extracted Text")

    st.text_area(
        "Detected text",
        extracted_text,
        height=250
    )

    st.success("Travel information extracted successfully.")