import streamlit as st
import os
import docx
import PyPDF2
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import openai
import numpy as np
import pandas as pd

# 🔐 Φόρτωση OpenAI API Key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 🧾 Τίτλος + Δήλωση απορρήτου
st.title("🤖 CV Matcher με OpenAI Embeddings")

st.info("""
🔒 **Πληροφορίες Ασφαλείας / Προστασίας Προσωπικών Δεδομένων**

Η παρούσα εφαρμογή χρησιμοποιεί την πλατφόρμα Streamlit για την ανάλυση βιογραφικών (CV) τοπικά, χωρίς αποθήκευση δεδομένων.

✅ Τα αρχεία που ανεβάζετε (PDF, DOCX, TXT):
- **ΔΕΝ αποθηκεύονται** στο cloud ή σε κάποιον διακομιστή.
- **ΔΕΝ κοινοποιούνται** σε τρίτους.
- **Χρησιμοποιούνται μόνο προσωρινά** για την εκτέλεση της αξιολόγησης.

🧠 Η ανάλυση γίνεται με χρήση του GPT API της OpenAI, και το περιεχόμενο αποστέλλεται κρυπτογραφημένα και μόνο για την παροχή απάντησης.

🕒 Μετά τη λήξη της συνεδρίας (π.χ. αν κλείσετε τη σελίδα), όλα τα δεδομένα διαγράφονται αυτόματα από τη μνήμη.
""")

# 📤 Upload CVs
uploaded_files = st.file_uploader("📤 Ανέβασε έως 20 βιογραφικά (PDF, DOCX, TXT)", accept_multiple_files=True, type=["pdf", "txt", "docx"])

# 📝 Περιγραφή και ερώτηση
job_description = st.text_area("📝 Περιγραφή Θέσης", placeholder="Π.χ. Μηχανικός Τηλεπικοινωνιών με εμπειρία σε AutoCAD και Οπτικές Ίνες")
user_question = st.text_input("🤔 Ρώτησε κάτι για τα CV (π.χ. 'ποιος έχει εμπειρία σε AutoCAD;')")

# 📎 Κουμπί Υποβολής
submitted = st.button("🔍 Ανάλυση Βιογραφικών")

# ⚠️ Όριο 20 αρχείων
if uploaded_files and len(uploaded_files) > 20:
    st.warning("⚠️ Μπορείτε να ανεβάσετε έως 20 αρχεία κάθε φορά.")
    st.stop()


def extract_text(file):
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    elif file.type == "text/plain":
        return str(file.read(), "utf-8")
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    return ""


def get_embedding(text):
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return np.array(response.data[0].embedding)


# 🚀 Ανάλυση μόνο όταν πατηθεί το κουμπί
if submitted and uploaded_files and job_description:
    with st.spinner("🔍 Γίνεται ανάλυση..."):
        job_embed = get_embedding(job_description)

        results = []
        for file in uploaded_files:
            content = extract_text(file)
            if content.strip() == "":
                continue
            cv_embed = get_embedding(content[:3000])  # OpenAI limit
            similarity = cosine_similarity([job_embed], [cv_embed])[0][0]
            results.append((file.name, similarity))

        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

        # 🔽 Εξαγωγή σε CSV
        if sorted_results:
            export_data = []
            for name, score in sorted_results:
                row = {"Όνομα αρχείου": name, "Σκορ Ταύτισης (%)": round(score * 100, 2)}
                export_data.append(row)

            df = pd.DataFrame(export_data)
            csv = df.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="💾 Εξαγωγή αποτελεσμάτων σε CSV",
                data=csv,
                file_name='cv_match_results.csv',
                mime='text/csv',
            )

        # 📊 Εμφάνιση αποτελεσμάτων
        st.subheader("📊 Κατάταξη υποψηφίων:")
        for name, score in sorted_results:
            st.markdown(f"### 📄 {name}")
            st.markdown(f"**Σκορ Ταύτισης:** {round(score * 100, 2)}%")

            for file in uploaded_files:
                if file.name == name:
                    content = extract_text(file)

                    if user_question:
                        with st.spinner(f"🤖 Επεξεργασία ερώτησης για {name}..."):
                            response = openai.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {"role": "system", "content": "Είσαι βοηθός πρόσληψης. Δώσε σύντομη απάντηση ναι ή όχι."},
                                    {"role": "user", "content": f"Ακολουθεί ένα βιογραφικό:\n{content[:3000]}\n\nΕρώτηση: {user_question}"}
                                ]
                            )
                            reply = response.choices[0].message.content.strip()
                            st.info(f"GPT: {reply}")

                    st.expander("📖 Δες περιεχόμενο βιογραφικού").write(content[:2000])
                    break
