
import os
import json
import gradio as gr
import chromadb

from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. PATHS AND CONFIGURATION
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(BASE_DIR, "sports_chroma_db")
IMAGE_DIR = os.path.join(BASE_DIR, "sports_images")
CATALOG_PATH = os.path.join(BASE_DIR, "image_catalog.json")

DEVICE = "cpu"


# --------------------------------------------------
# 2. LOAD CHROMADB AND EMBEDDING MODEL
# --------------------------------------------------

print("Loading document embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=DEVICE
)

client = chromadb.PersistentClient(path=DB_PATH)

collection = client.get_collection(
    name="sports_documents_collection"
)

print("ChromaDB loaded successfully!")


# --------------------------------------------------
# 3. LOAD IMAGE CATALOGUE
# --------------------------------------------------

with open(CATALOG_PATH, "r") as file:
    sports_image_catalog = json.load(file)

# Use the same sorted file order as the original catalogue
image_files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

sports_image_paths = {
    category: [
        os.path.join(IMAGE_DIR, image_files[number - 1])
        for number in numbers
        if 1 <= number <= len(image_files)
    ]
    for category, numbers in sports_image_catalog.items()
}


# --------------------------------------------------
# 4. IMAGE SEARCH
# --------------------------------------------------

def search_sports_images(query, top_k=3):

    q = query.lower()

    category_keywords = [
        ("sports department office",
         ["sports department office", "sports office"]),
        ("indoor table tennis",
         ["indoor tennis", "table tennis", "indoor games"]),
        ("outdoor tennis",
         ["outdoor tennis", "tennis"]),
        ("badminton", ["badminton"]),
        ("cricket", ["cricket"]),
        ("basketball", ["basketball", "basket"]),
        ("football", ["football"]),
        ("carrom", ["carrom"])
    ]

    selected_category = None

    for category, keywords in category_keywords:
        if any(keyword in q for keyword in keywords):
            selected_category = category
            break

    if selected_category is None:
        return []

    return sports_image_paths[selected_category][:top_k]


# --------------------------------------------------
# 6. QUERY INTENT
# --------------------------------------------------

def detect_query_intent(query):

    q = query.lower().strip()

    if q in ["who are you", "who are you?"]:
        return "identity"

    if q in ["who am i", "who am i?"]:
        return "user_identity"

    image_words = [
        "image", "images", "picture",
        "pictures", "photo", "photos"
    ]

    image_actions = [
        "show", "display", "find",
        "get", "give", "view", "see", "fetch"
    ]

    if (
        any(word in q for word in image_words)
        and any(action in q for action in image_actions)
    ):
        return "text_and_image"

    personal_keywords = [
        "my age", "my birthday", "my address",
        "my phone number", "my password",
        "my family", "my personal life",
        "where do i live", "what is my name"
    ]

    if any(keyword in q for keyword in personal_keywords):
        return "personal"

    return "text_only"


# --------------------------------------------------
# 7. JUNE 2026 ACTIVITY PLANNER
# --------------------------------------------------

june_2026_activities = [
    {"date": "May & June 2026",
     "activity": "Faculty Sports Event"},
    {"date": "3 June 2026",
     "activity": "World Bicycle Day"},
    {"date": "5 June 2026",
     "activity": "World Environment Day digital campaign and observation"},
    {"date": "15 June 2026",
     "activity": "Sports Quota Admissions - selections"},
    {"date": "19 June 2026",
     "activity": "International Yoga Day celebration"},
    {"date": "21 June 2026",
     "activity": "International Day of Yoga - demonstrations and seminars"},
    {"date": "24 June 2026",
     "activity": "Sports Quota Admissions - selections"},
    {"date": "26 June 2026",
     "activity": "International Day Against Drug Abuse and Illicit Trafficking"},
    {"date": "June 2026",
     "activity": "Talent Hunt, university team formation, club registration and sports selection trials"}
]


def get_planner_answer(query):

    q = query.lower()

    if "june" in q and "2026" in q:

        lines = [
            "Here are the activities listed for June 2026:"
        ]

        for item in june_2026_activities:
            lines.append(
                f"• {item['date']}: {item['activity']}"
            )

        return "\n".join(lines)

    return None


# --------------------------------------------------
# 8. DOCUMENT RETRIEVAL
# --------------------------------------------------

def retrieve_documents(query, top_k=3):

    query_embedding = embedding_model.encode(
        query
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    docs = []

    for text, metadata in zip(
        results["documents"][0],
        results["metadatas"][0]
    ):

        docs.append({
            "text": text,
            "source": metadata["source"],
            "page": metadata["page"]
        })

    return docs


# --------------------------------------------------
# 9. FAST DOCUMENT ANSWERING
# --------------------------------------------------

def fast_document_answer(query):

    docs = retrieve_documents(query, top_k=5)

    if not docs:
        return (
            "I couldn't find relevant information in the "
            "university documents."
        )

    # Retrieve distances directly from ChromaDB
    query_embedding = embedding_model.encode(query).tolist()

    raw = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )

    parts = []
    sources = set()

    # Keep reasonably relevant passages
    for text, metadata, distance in zip(
        raw["documents"][0],
        raw["metadatas"][0],
        raw["distances"][0]
    ):

        if distance > 0.72:
            continue

        text = text.strip()

        if text and text not in parts:
            parts.append(text)
            sources.add(
                f"{metadata['source']} (page {metadata['page']})"
            )

    if not parts:
        return (
            "I couldn't find sufficiently relevant information "
            "in the university documents."
        )

    return (
        "\\n\\n".join(parts)
        + "\\n\\nSources:\\n"
        + "\\n".join(f"• {source}" for source in sorted(sources))
    )


# 10. CHATBOT RESPONSE
# --------------------------------------------------

def generate_chatbot_response(query, chat_history=None):

    query = query.strip()
    intent = detect_query_intent(query)

    planner_answer = get_planner_answer(query)
    if planner_answer:
        return planner_answer, []

    if intent == "identity":
        return (
            "I am the Presidency University Sports and Student "
            "Activities Assistant. I help students with sports, "
            "events, clubs, activities, and university information."
        ), []

    if intent == "user_identity":
        return (
            "I don't know your identity unless you share details "
            "with me in this conversation."
        ), []

    if intent == "personal":
        return (
            "I don't have access to your personal information. "
            "I can only help with university sports and student activities."
        ), []

    if intent == "text_and_image":
        images = search_sports_images(query, top_k=3)

        if not images:
            return (
                "I couldn't find images for that sports category.",
                []
            )

        # Image-only request: return images without unrelated document text.
        image_words = ["image", "images", "picture", "pictures", "photo", "photos"]
        text_words = ["explain", "describe", "tell me about", "information about"]

        asks_for_text = any(word in query.lower() for word in text_words)

        if not asks_for_text:
            return "Here are the requested sports images:", images

        answer = fast_document_answer(query)
        return answer, images

    answer = fast_document_answer(query)
    return answer, []


# 11. GRADIO CHAT INTERFACE
# --------------------------------------------------

def respond(message, history):

    if not message or not message.strip():
        return history, ""

    history = history or []

    history.append({
        "role": "user",
        "content": message
    })

    answer, images = generate_chatbot_response(
        message,
        chat_history=history
    )

    history.append({
        "role": "assistant",
        "content": answer
    })

    for image_path in images:

        history.append({
            "role": "assistant",
            "content": gr.Image(
                value=image_path,
                height=250,
                show_label=False
            )
        })

    return history, ""


def clear_chat():
    return [], ""


# --------------------------------------------------
# 12. BUILD INTERFACE
# --------------------------------------------------

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Presidency University Sports Assistant"
) as demo:

    gr.Markdown("""
    # Presidency University
    ### Sports & Student Activities Assistant

    Ask me about university sports, events, clubs and activities.
    """)

    chatbot = gr.Chatbot(
        label="Conversation",
        height=500,
        layout="bubble"
    )

    with gr.Row():

        message = gr.Textbox(
            placeholder="Ask your question here...",
            show_label=False,
            scale=8
        )

        send = gr.Button(
            "Send",
            variant="primary"
        )

    with gr.Row():

        clear = gr.Button("New Chat")

        theme_choice = gr.Radio(
            ["Light", "Dark"],
            value="Light",
            label="Appearance"
        )

    send.click(
        respond,
        inputs=[message, chatbot],
        outputs=[chatbot, message]
    )

    message.submit(
        respond,
        inputs=[message, chatbot],
        outputs=[chatbot, message]
    )

    clear.click(
        clear_chat,
        outputs=[chatbot, message]
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), theme=gr.themes.Soft())
 