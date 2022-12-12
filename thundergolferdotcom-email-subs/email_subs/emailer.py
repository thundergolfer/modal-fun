import base64
from email.message import EmailMessage

from typing import Protocol


class EmailSender(Protocol):
    def send(self, message: EmailMessage) -> None:
        ...


class GmailSender(EmailSender):
    def __init__(self, creds) -> None:
        from googleapiclient.discovery import build

        self.service = build("gmail", "v1", credentials=creds)

    def send(self, message: EmailMessage) -> None:
        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        raw_encoded_message = {"raw": encoded_message}
        # pylint: disable=E1101
        send_message = (
            self.service.users()
            .messages()
            .send(userId="me", body=raw_encoded_message)
            .execute()
        )
        print(f'Message Id: {send_message["id"]}')
        return None


def send(
    *,
    sender: EmailSender,
    subject: str,
    content: str,
    from_addr: str,
    recipient: str,
) -> None:
    message = EmailMessage()
    message.set_content(content)
    message["To"] = recipient,
    message["From"] = from_addr
    message["Subject"] = subject
    sender.send(message)
