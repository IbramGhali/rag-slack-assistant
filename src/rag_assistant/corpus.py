"""Synthetic policy corpus for the fictional Northwind Learning.

Every document is defined here and written to Markdown by `rag generate`, so the corpus is
reproducible and contains no real organisation's material. Four policies have a superseded
earlier version with conflicting facts, to test that retrieval prefers the current one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Doc:
    doc_id: str
    title: str
    area: str
    effective: str
    sections: tuple[tuple[str, str], ...]
    superseded_by: str | None = None


DOCS: list[Doc] = [
    Doc(
        "payment-plans",
        "Payment Plans",
        "Payments",
        "2025-01-15",
        (
            (
                "Eligibility",
                "Any student enrolling in a course priced over $1,500 can apply for a "
                "payment plan. Applications are assessed at enrolment and approved within two "
                "business days.",
            ),
            (
                "Instalment options",
                "Plans run for 6, 12 or 18 monthly instalments. The first "
                "instalment is due on the course start date, and later instalments are due on the "
                "same day each month.",
            ),
            (
                "Missed payments",
                "If an instalment is missed, the student receives a reminder after "
                "3 days. Course access is paused after 21 days without payment and restored within "
                "one business day of the overdue amount being paid.",
            ),
            (
                "Changing a plan",
                "A student can change the plan length once per course by contacting "
                "Student Accounts. Changes take effect from the next instalment date.",
            ),
        ),
    ),
    Doc(
        "refunds-2024",
        "Refund Policy (2024)",
        "Payments",
        "2024-01-01",
        (
            ("Refund window", "Students may request a full refund within 7 days of enrolment."),
            (
                "Processing time",
                "Approved refunds are paid within 15 business days to the original payment method.",
            ),
            ("Exceptions", "No exceptions are made after the refund window closes."),
        ),
        superseded_by="refunds",
    ),
    Doc(
        "refunds",
        "Refund Policy",
        "Payments",
        "2025-07-01",
        (
            (
                "Refund window",
                "Students may request a full refund within 14 days of enrolment, "
                "provided they have completed less than 10 percent of the course.",
            ),
            (
                "Processing time",
                "Approved refunds are paid within 10 business days to the original payment method.",
            ),
            (
                "Partial refunds",
                "After the refund window, refunds are calculated pro rata for units "
                "the student has not started.",
            ),
            (
                "Exceptions",
                "Students with a documented medical condition or bereavement can apply "
                "for a refund outside the window. Supporting documents must be supplied within 30 "
                "days.",
            ),
        ),
    ),
    Doc(
        "late-fees",
        "Late Fees",
        "Payments",
        "2025-01-15",
        (
            (
                "Late fee amount",
                "A late fee of $25 is charged when an instalment is more than 5 days overdue.",
            ),
            ("Limits", "No more than 3 late fees are charged on a single payment plan."),
            (
                "Hardship waiver",
                "Students experiencing financial hardship can ask Student Accounts "
                "to waive late fees. Waivers are decided within 5 business days.",
            ),
        ),
    ),
    Doc(
        "scholarships",
        "Scholarships",
        "Payments",
        "2025-02-01",
        (
            ("Merit scholarship", "Merit scholarships reduce the course fee by up to 20 percent."),
            (
                "How to apply",
                "Applications close 30 days before the course start date and must "
                "include a personal statement of no more than 500 words.",
            ),
            (
                "Limits",
                "A student can hold only one scholarship at a time, and scholarships cannot "
                "be combined with sales discounts.",
            ),
        ),
    ),
    Doc(
        "course-deferral",
        "Course Deferral",
        "Enrolment",
        "2025-01-15",
        (
            (
                "Deferral period",
                "Students can defer the start of a course by up to 12 months. Each "
                "enrolment can be deferred only once.",
            ),
            (
                "How to request",
                "Deferral requests must be lodged at least 10 business days before "
                "the course start date through the student portal.",
            ),
        ),
    ),
    Doc(
        "withdrawal",
        "Withdrawing From a Course",
        "Enrolment",
        "2025-01-15",
        (
            (
                "How to withdraw",
                "Students withdraw through the student portal. Withdrawal is "
                "effective from the date the request is submitted.",
            ),
            (
                "Census date",
                "Withdrawals after the census date, which is 28 days after the course "
                "start, are not eligible for a refund.",
            ),
            (
                "Academic record",
                "A withdrawn unit appears on the academic record with a grade of W "
                "and does not affect the grade average.",
            ),
        ),
    ),
    Doc(
        "credit-transfer",
        "Credit Transfer",
        "Enrolment",
        "2025-03-01",
        (
            (
                "Maximum credit",
                "Students can receive credit for up to 50 percent of the units in a course.",
            ),
            (
                "Evidence",
                "Credit requires certified transcripts for study completed in the last 5 "
                "years. Older study is assessed case by case.",
            ),
        ),
    ),
    Doc(
        "identity-verification",
        "Identity Verification",
        "Enrolment",
        "2025-01-15",
        (
            (
                "Requirements",
                "Every student must verify their identity before enrolment with two "
                "forms of identification, one of which must include a photo.",
            ),
            (
                "Remote verification",
                "Students who cannot attend in person can complete a recorded "
                "video verification call with the Admissions team.",
            ),
        ),
    ),
    Doc(
        "annual-leave-2024",
        "Annual Leave (2024)",
        "People",
        "2024-01-01",
        (
            ("Entitlement", "Full-time staff receive 20 days of annual leave per year."),
            ("Requests", "Leave requests must be submitted 2 weeks in advance."),
            ("Carry over", "Unused annual leave cannot be carried over to the next year."),
        ),
        superseded_by="annual-leave",
    ),
    Doc(
        "annual-leave",
        "Annual Leave",
        "People",
        "2025-07-01",
        (
            (
                "Entitlement",
                "Full-time staff receive 22 days of annual leave per year. Part-time "
                "staff receive a pro rata entitlement.",
            ),
            (
                "Requests",
                "Leave requests must be submitted at least 10 business days in advance "
                "through the HR system and approved by the staff member's manager.",
            ),
            (
                "Carry over",
                "Up to 5 days of unused annual leave can be carried over to the next year.",
            ),
        ),
    ),
    Doc(
        "sick-leave",
        "Sick Leave",
        "People",
        "2025-01-15",
        (
            ("Entitlement", "Staff receive 10 days of paid sick leave per year."),
            (
                "Evidence",
                "A medical certificate is required for absences of more than 2 consecutive days.",
            ),
        ),
    ),
    Doc(
        "parental-leave",
        "Parental Leave",
        "People",
        "2025-01-15",
        (
            ("Primary carer", "Primary carers receive 16 weeks of paid parental leave."),
            ("Secondary carer", "Secondary carers receive 4 weeks of paid parental leave."),
            ("Eligibility", "Staff are eligible after 12 months of continuous service."),
        ),
    ),
    Doc(
        "remote-work",
        "Remote Work",
        "People",
        "2025-04-01",
        (
            (
                "Arrangement",
                "Staff can work remotely up to 3 days per week with their manager's approval.",
            ),
            (
                "Security",
                "Remote work must use a secure home network or the company VPN. Public "
                "wifi is not allowed for confidential data.",
            ),
        ),
    ),
    Doc(
        "expenses-2024",
        "Expense Claims (2024)",
        "People",
        "2024-01-01",
        (
            ("Deadline", "Expense claims must be submitted within 60 days of purchase."),
            ("Meals", "Meal expenses are capped at $40 per day."),
            ("Approval", "Claims over $300 need approval from a department head."),
        ),
        superseded_by="expenses",
    ),
    Doc(
        "expenses",
        "Expense Claims",
        "People",
        "2025-07-01",
        (
            (
                "Deadline",
                "Expense claims must be submitted within 30 days of purchase, with "
                "itemised receipts.",
            ),
            ("Meals", "Meal expenses are capped at $60 per day when travelling for work."),
            ("Approval", "Claims over $500 need approval from the staff member's manager."),
        ),
    ),
    Doc(
        "travel",
        "Business Travel",
        "People",
        "2025-01-15",
        (
            ("Booking", "All travel must be booked through the company travel portal."),
            (
                "Flights",
                "Economy class is used for flights under 6 hours. Longer flights can be "
                "booked in premium economy.",
            ),
            ("Approval", "Travel must be approved at least 14 days before departure."),
        ),
    ),
    Doc(
        "passwords",
        "Password and Login Standard",
        "IT",
        "2025-02-01",
        (
            ("Password length", "Passwords must be at least 14 characters long."),
            (
                "Multi-factor authentication",
                "Multi-factor authentication is required for email, "
                "the student system and remote access.",
            ),
            (
                "Reuse and rotation",
                "The last 10 passwords cannot be reused. Passwords are changed "
                "only when a compromise is suspected, not on a fixed schedule.",
            ),
        ),
    ),
    Doc(
        "device-security",
        "Device Security",
        "IT",
        "2025-02-01",
        (
            ("Encryption", "All company laptops use full-disk encryption."),
            ("Screen lock", "Screens lock automatically after 5 minutes of inactivity."),
            (
                "Lost devices",
                "A lost or stolen device must be reported to the IT service desk "
                "within 1 hour so it can be wiped remotely.",
            ),
        ),
    ),
    Doc(
        "data-classification",
        "Data Classification",
        "IT",
        "2025-02-01",
        (
            ("Levels", "Data is classified as public, internal, confidential or restricted."),
            ("Student data", "Student personal information is classified as confidential."),
            (
                "Restricted data",
                "Restricted data, such as payment card details, must be encrypted "
                "at rest and in transit and must never be sent by email.",
            ),
        ),
    ),
    Doc(
        "incident-reporting",
        "Security Incident Reporting",
        "IT",
        "2025-02-01",
        (
            (
                "When to report",
                "Report any suspected security incident, including phishing emails "
                "you clicked, within 1 hour.",
            ),
            (
                "How to report",
                "Contact the security desk on extension 4400 or at "
                "security@northwind.example. Do not try to investigate the incident yourself.",
            ),
        ),
    ),
    Doc(
        "software-requests",
        "Software Requests",
        "IT",
        "2025-02-01",
        (
            (
                "How to request",
                "Request new software through the IT portal. Requests are reviewed "
                "within 5 business days.",
            ),
            ("Licences", "Personal software licences must not be used for company work."),
        ),
    ),
    Doc(
        "complaints",
        "Student Complaints",
        "Student Services",
        "2025-03-01",
        (
            (
                "Lodging a complaint",
                "Students can lodge a complaint within 30 days of the issue "
                "through the student portal.",
            ),
            (
                "Timeframes",
                "Complaints are acknowledged within 2 business days and resolved within "
                "20 business days.",
            ),
            (
                "Escalation",
                "If the student is not satisfied, they can escalate to the independent "
                "education ombudsman.",
            ),
        ),
    ),
    Doc(
        "assessment-extensions",
        "Assessment Extensions",
        "Student Services",
        "2025-03-01",
        (
            (
                "Short extensions",
                "Students can receive an extension of up to 7 days with supporting "
                "evidence. Requests must be made before the due date.",
            ),
            (
                "Longer extensions",
                "Extensions longer than 7 days require a special consideration application.",
            ),
        ),
    ),
    Doc(
        "accessibility",
        "Accessibility and Learning Support",
        "Student Services",
        "2025-03-01",
        (
            (
                "Learning access plans",
                "Students with a disability or health condition can request "
                "a learning access plan that sets out reasonable adjustments.",
            ),
            (
                "Timing",
                "Students should register at least 4 weeks before their first assessment so "
                "adjustments are in place.",
            ),
        ),
    ),
    Doc(
        "academic-integrity",
        "Academic Integrity",
        "Student Services",
        "2025-03-01",
        (
            (
                "Generative AI",
                "Generative AI tools can be used in an assessment only when the "
                "assessment instructions say so, and their use must be acknowledged.",
            ),
            (
                "Breaches",
                "A first minor breach results in a formal warning and a resubmission. "
                "Serious or repeated breaches are referred to the academic integrity panel.",
            ),
        ),
    ),
    Doc(
        "call-recording",
        "Sales Call Recording",
        "Sales",
        "2025-01-15",
        (
            (
                "Disclosure",
                "Agents must tell the customer at the start of every sales call that the "
                "call is recorded.",
            ),
            ("Retention", "Sales call recordings are kept for 2 years and then deleted."),
        ),
    ),
    Doc(
        "discount-approval-2024",
        "Sales Discount Approval (2024)",
        "Sales",
        "2024-01-01",
        (
            ("Agent limit", "Sales agents can offer a discount of up to 10 percent."),
            (
                "Higher discounts",
                "Discounts above 10 percent need manager approval, up to a maximum of 20 percent.",
            ),
        ),
        superseded_by="discount-approval",
    ),
    Doc(
        "discount-approval",
        "Sales Discount Approval",
        "Sales",
        "2025-07-01",
        (
            ("Agent limit", "Sales agents can offer a discount of up to 15 percent."),
            (
                "Higher discounts",
                "Discounts above 15 percent need team lead approval, up to a "
                "maximum of 25 percent.",
            ),
            ("Combining offers", "Discounts cannot be combined with scholarships."),
        ),
    ),
    Doc(
        "records-retention",
        "Records Retention",
        "Compliance",
        "2025-02-01",
        (
            (
                "Student records",
                "Student academic records are kept for 7 years after the student "
                "completes or leaves the course.",
            ),
            (
                "Deletion requests",
                "Requests to delete personal information are assessed by the "
                "privacy officer, because some records must be kept by law.",
            ),
        ),
    ),
    Doc(
        "privacy",
        "Privacy",
        "Compliance",
        "2025-02-01",
        (
            (
                "Collection",
                "Northwind Learning collects only the personal information needed to "
                "deliver and administer courses.",
            ),
            (
                "Access requests",
                "Requests for access to personal information are answered within 30 days.",
            ),
            (
                "Contact",
                "Privacy questions go to the privacy officer at privacy@northwind.example.",
            ),
        ),
    ),
]


def render(doc: Doc) -> str:
    front = [
        "---",
        f"doc_id: {doc.doc_id}",
        f"title: {doc.title}",
        f"area: {doc.area}",
        f"effective: {doc.effective}",
        f"superseded_by: {doc.superseded_by or ''}",
        "---",
        "",
        f"# {doc.title}",
        "",
    ]
    body = []
    for heading, text in doc.sections:
        body += [f"## {heading}", "", text, ""]
    return "\n".join(front + body)


def write_corpus(directory: Path) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob("*.md"):
        old.unlink()
    for doc in DOCS:
        (directory / f"{doc.doc_id}.md").write_text(render(doc), encoding="utf-8", newline="\n")
    return len(DOCS)
