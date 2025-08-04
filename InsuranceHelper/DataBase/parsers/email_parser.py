import email

def extract_text_from_eml(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        msg = email.message_from_file(f)

    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    body += payload.decode('utf-8')
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            body = payload.decode('utf-8')

    return body
