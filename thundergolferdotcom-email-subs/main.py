import modal
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


stub = modal.Stub(name="thundergolferdotcom-email-subs")
stub.confirmation_code_to_email = modal.Dict()
web_app = FastAPI()

# Check relatively frequently for new posts, because subscribers should
# be among first to hear of a new post.
@stub.function(schedule=modal.Period(hours=3))
def check_for_new_post():
    # If new, unseen post:
    # 1. mark post as seen
    # 2. send out emails to all confirmed subscribers
    pass

@stub.function
def send_confirmation_email(email: str):
    pass 

@web_app.get("/confirm")
def confirm(email: str, code: str):
    pass

@web_app.get("/unsubscribe")
def unsubscribe(email: str, code: str):
    # Check code against email. If match, unsubscribe user
    # and send back HTML page showing them they were unsubscribed.
    pass

@web_app.get("/subscribe")
def subscribe(email: str):
    # 1. check if email is already subscribed
    # 2. send confirmation email if not
    send_confirmation_email.spawn(email="")
    return {
        "hello": "world"
    }


@stub.asgi
def web():
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://thundergolfer.com",
            "https://thundergolfer.com",
            "http://localhost:4000",
            "http://localhost:4000/",
            "localhost:4000",
            "localhost:4000/",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return web_app


if __name__ == "__main__":
    stub.serve()
