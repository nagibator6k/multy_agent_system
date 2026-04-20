def route(text):
    text = text.lower()
    if "объясни" in text or "что такое" in text:
        return "tutor"
    return "assessment"