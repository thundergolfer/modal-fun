"""
blog2email.py has the sole function to taking blog post information
and creating an email notification subject and body.
"""
from typing import NamedTuple


class EmailCopy(NamedTuple):
    subject: str
    body: str


def construct_new_blogpost_email(
    blog_url: str,
    blog_name: str,
    blog_titles: list[str],
    blog_links: list[str],
    unsubscribe_link: str,
    twitter_url: str,
) -> EmailCopy:
    main_body = "\n".join(
        [
            "Hey,",
            f"Thanks for subscribing to <a href='{blog_url}'>{blog_name}</a>, I've got a new blog post for you!"
            "",
            f"<strong><em><a href='{blog_links[0]}'>{blog_titles[0]}</a></em></strong>",
            "<br><br>",
            "If you like it or have any feedback, I'm always happy to hear from you. "
            f"I'm reachable on <a href='{twitter_url}'>Twitter</a> and at this email address.",
        ]
    )

    if len(blog_titles) > 1:
        # TODO: Handle this case by creating an unordered list of blog post links.
        bonus_part = ""
    else:
        bonus_part = ""

    unsub_part = (
        "<br><br>"
        "If you'd like to stop receiving these emails, "
        f"<a href='{unsubscribe_link}'>click here to unsubscribe</a>."
    )

    return EmailCopy(
        subject=f"New post '{blog_titles[0]}'",
        body=(main_body + "\n" + bonus_part + "\n" + unsub_part),
    )


def confirm_subscription_email(
    blog_name: str, blog_url: str, confirmation_link: str
) -> EmailCopy:
    return EmailCopy(
        subject=f"Confirm subscription to {blog_name} blog",
        body=(
            f"To confirm your subscription to <a href='{blog_url}/'>{blog_url}</a>, "
            f"please <strong><a href='{confirmation_link}'>click here</a></strong>.\n\n"
            f"If you didn't subscribe to <a href='{blog_url}'>{blog_name}</a>, "
            "please disregard this email.\n"
        ),
    )
