"""
blog2email.py has the sole function to taking blog post information
and creating an email notification subject and body.
"""
from typing import NamedTuple

class EmailCopy(NamedTuple):
    subject: str
    body: str

def construct_new_blogpost_email(
    blog_titles: list[str],
    blog_links: list[str],
    unsubscribe_link: str,
) -> EmailCopy:
    twitter_url = "https://twitter.com/jonobelotti_IO"
    main_body = "\n".join([
        "Hey,",
        "Thanks for subscribing to <a href=''>thundergolfer.com/blog</a>, I've got a new blog post for you!"
        "",
        f"<strong><em><a href='{blog_links[0]}'>{blog_titles[0]}</a></em></strong>",
        "",
        "If you like it or have any feedback, I'm always happy to hear from you."
        f"I'm reachable on <a href='{twitter_url}'>Twitter</a> and at this email address."
    ])

    if len(blog_titles) > 1:
        # TODO: Handle this case by creating an unordered list of blog post links.
        bonus_part = ""
    else:
        bonus_part = ""

    unsub_part = (
        "If you'd like to stop receiving these emails, "
        f"<a href='{unsubscribe_link}'>click here to unsubscribe</a>."
    )

    return EmailCopy(
        subject=f"New post '{blog_titles[0]}'",
        body=(main_body + "\n" + bonus_part + "\n" + unsub_part),
    )

def confirm_subscription_email(confirmation_link: str) -> EmailCopy:
    return EmailCopy(
        subject="Confirm subscription to thundergolfer.com blog",
        body=(
            "To confirm your subscription to <a href='https://thundergolfer.com/'>thundergolfer.com/blog</a>, "
            f"please <strong><a href='{confirmation_link}'>click here</a></strong>.\n\n"
            "If you didn't subscribe to <a href='https://thundergolfer.com/'>thundergolfer.com/blog</a>, "
            "please disregard this email.\n"
        )
    )
