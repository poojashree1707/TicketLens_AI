import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes
from ocr import process_image


st.set_page_config(
    page_title="TravelDoc AI",
    page_icon="✈️",
    layout="wide"
)

st.title("TravelDoc AI")

st.subheader(
    "OCR-Based Travel Ticket Information Extractor"
)


uploaded_file = st.file_uploader(
    "Upload any travel ticket",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "tif",
        "tiff",
        "webp",
        "pdf"
    ]
)


if uploaded_file:

    if uploaded_file.type == "application/pdf":

        try:

            pages = convert_from_bytes(
                uploaded_file.read(),
                dpi=250
            )

        except Exception:

            st.error(
                "PDF could not be processed. Install Poppler and make sure it is added to PATH."
            )

            st.stop()

    else:

        pages = [
            Image.open(
                uploaded_file
            ).convert("RGB")
        ]


    all_ocr_text = []

    all_basic = []

    all_passengers = []


    for page_number, page in enumerate(
        pages,
        start=1
    ):

        text, basic, passengers = process_image(
            page
        )

        all_ocr_text.append(text)

        all_basic.append(basic)

        all_passengers.extend(passengers)


    final_basic = {}


    for key in all_basic[0].keys():

        value = "Not detected"

        for item in all_basic:

            if (
                item.get(key)
                and item.get(key) != "Not detected"
            ):

                value = item[key]

                break

        final_basic[key] = value


    unique_passengers = []

    seen = set()


    for passenger in all_passengers:

        key = (
            passenger["Name"].upper(),
            passenger["Age"],
            passenger["Gender"],
            passenger["Seat No"]
        )

        if key not in seen:

            seen.add(key)

            unique_passengers.append(
                passenger
            )


    st.success(
        "Ticket processed successfully."
    )


    st.header("Ticket Information")


    col1, col2 = st.columns(2)


    with col1:

        st.write(
            "Transport Type:",
            final_basic["Transport Type"]
        )

        st.write(
            "Ticket Number:",
            final_basic["Ticket Number"]
        )

        st.write(
            "PNR:",
            final_basic["PNR"]
        )

        st.write(
            "Journey Date:",
            final_basic["Journey Date"]
        )

        st.write(
            "Journey Time:",
            final_basic["Journey Time"]
        )

        st.write(
            "From:",
            final_basic["From"]
        )

        st.write(
            "To:",
            final_basic["To"]
        )


    with col2:

        st.write(
            "Operator / Travel Company:",
            final_basic["Operator / Travel Company"]
        )

        st.write(
            "Bus / Service:",
            final_basic["Bus / Service"]
        )

        st.write(
            "Boarding Point:",
            final_basic["Boarding Point"]
        )

        st.write(
            "Dropping Point:",
            final_basic["Dropping Point"]
        )

        st.write(
            "Ticket Price:",
            final_basic["Ticket Price"]
        )

        st.write(
            "Booking Name:",
            final_basic["Booking Name"]
        )

        st.write(
            "Seat / Coach:",
            final_basic["Seat / Coach"]
        )

        st.write(
            "Platform / Gate:",
            final_basic["Platform / Gate"]
        )


    st.header("Passenger Information")


    st.metric(
        "Passenger Count",
        len(unique_passengers)
    )


    if unique_passengers:

        for number, passenger in enumerate(
            unique_passengers,
            start=1
        ):

            st.subheader(
                f"Passenger {number}"
            )

            c1, c2, c3, c4 = st.columns(4)


            with c1:

                st.write("Name")

                st.write(
                    passenger["Name"]
                )


            with c2:

                st.write("Age")

                st.write(
                    passenger["Age"]
                )


            with c3:

                st.write("Gender")

                st.write(
                    passenger["Gender"]
                )


            with c4:

                st.write("Seat No")

                st.write(
                    passenger["Seat No"]
                )


    else:

        st.warning(
            "Passenger details could not be detected from this ticket."
        )


    with st.expander("View OCR Text"):

        st.text(
            "\n\n".join(all_ocr_text)
        )


else:

    st.info(
        "Upload a travel ticket image or PDF to extract the information."
    )
    