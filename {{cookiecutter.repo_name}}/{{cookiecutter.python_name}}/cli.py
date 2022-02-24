import base64
import click

import {{cookiecutter.python_name}}.core


def print_banner() -> None:
    """Prints a Colorful Logo
    """
    banner = "G1sxMDc7NDBtG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szOD" \
             "s1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20g" \
             "G1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20mG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szOD" \
             "s1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20g" \
             "G1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gChtbMzg7NTttIBtbMz" \
             "g7NTttIBtbMzg7NTttIBtbMzg7NTttIBtbMzg7NTttIBtbMzg7NTttIBtbMzg7NTttIBtbMzg7NTttIBtbMzg7NTttIBtbMzg7NTtt" \
             "IBtbMzg7NTttIBtbMzg7NTttIBtbMzg7NTttIBtbMzg7NTttIBtbMzg7NTttIBtbMzg7NTttKBtbMzg7NTttJRtbMzg7NTsyMjFtJR" \
             "tbMzg7NTsyMjFtJRtbMzg7NTsyMjFtJRtbMzg7NTsxODVtIxtbMzg7NTsxNzltIxtbMzg7NTsxNzltIxtbMzg7NTsxNzltIxtbMzg7" \
             "NTttJRtbMzg7NTttLBtbMzg7NTttIBtbMzg7NTttIBtbMzg7NTttIBtbMzg7NTttIBtbMzg7NTttIBtbMzg7NTttIBtbMzg7NTttIB" \
             "tbMzg7NTttIBtbMzg7NTttIBtbMzg7NTttIBtbMzg7NTttIBtbMzg7NTttIBtbMzg7NTttIBtbMzg7NTttIAobWzM4OzU7bSAbWzM4" \
             "OzU7bSAbWzM4OzU7bSAbWzM4OzU7bSAbWzM4OzU7bSAbWzM4OzU7bSAbWzM4OzU7bSAbWzM4OzU7bSAbWzM4OzU7bSAbWzM4OzU7bS" \
             "AbWzM4OzU7bSAbWzM4OzU7bSYbWzM4OzU7bSUbWzM4OzU7MjIxbSUbWzM4OzU7MjIxbSUbWzM4OzU7MjIxbSUbWzM4OzU7MjIxbSUb" \
             "WzM4OzU7MjIxbSUbWzM4OzU7MjIxbSUbWzM4OzU7MjIxbSUbWzM4OzU7MTg1bSMbWzM4OzU7MTc5bSMbWzM4OzU7MTc5bSMbWzM4Oz" \
             "U7MTc5bSMbWzM4OzU7MTc5bSMbWzM4OzU7MTc5bSMbWzM4OzU7MTc5bSMbWzM4OzU7MTc5bSMbWzM4OzU7bSMbWzM4OzU7bSobWzM4" \
             "OzU7bSAbWzM4OzU7bSAbWzM4OzU7bSAbWzM4OzU7bSAbWzM4OzU7bSAbWzM4OzU7bSAbWzM4OzU7bSAbWzM4OzU7bSAbWzM4OzU7bS" \
             "AbWzM4OzU7bSAKG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20sG1sz" \
             "ODs1O20jG1szODs1OzEwN20oG1szODs1OzEwN20oG1szODs1OzE0OW0jG1szODs1OzE4NW0jG1szODs1OzE4NW0jG1szODs1OzE4NW" \
             "0jG1szODs1OzIyMW0lG1szODs1OzIyMW0lG1szODs1OzIyMW0lG1szODs1OzIyMW0lG1szODs1OzIyMW0lG1szODs1OzIyMW0lG1sz" \
             "ODs1OzIyMW0jG1szODs1OzE3OW0jG1szODs1OzE3OW0jG1szODs1OzE3OW0jG1szODs1OzE3OW0jG1szODs1OzE3OW0jG1szODs1Oz" \
             "E3OW0jG1szODs1OzE3OW0jG1szODs1OzE3OW0jG1szODs1OzE3OW0jG1szODs1OzE3OW0jG1szODs1OzE3OW0jG1szODs1OzE3OW0j" \
             "G1szODs1O20jG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gChtbMzg7NTttIBtbMz" \
             "g7NTttIBtbMzg7NTttJRtbMzg7NTttKBtbMzg7NTsxMDdtKBtbMzg7NTsxMDdtKBtbMzg7NTsxMDdtKBtbMzg7NTsxMDdtKBtbMzg7" \
             "NTsxMDdtKBtbMzg7NTsxMDdtKBtbMzg7NTsxNDltIxtbMzg7NTsxODVtIxtbMzg7NTsxODVtIxtbMzg7NTsxODVtIxtbMzg7NTsxOD" \
             "VtIxtbMzg7NTsxODVtIxtbMzg7NTsxODVtIxtbMzg7NTsxODVtIxtbMzg7NTsyMjFtJRtbMzg7NTsyMjFtJRtbMzg7NTsyMjFtIxtb" \
             "Mzg7NTsxNzltIxtbMzg7NTsxNzltIxtbMzg7NTsxNzltIxtbMzg7NTsxNzltIxtbMzg7NTsxNzltIxtbMzg7NTsxNzltIxtbMzg7NT" \
             "sxNzltIxtbMzg7NTsxNzltIxtbMzg7NTsxNzltIxtbMzg7NTsxNzltIxtbMzg7NTsxNzltIxtbMzg7NTsxNzltIxtbMzg7NTsxNzlt" \
             "IxtbMzg7NTsxNzltIxtbMzg7NTsxNzltIxtbMzg7NTsxNzltIxtbMzg7NTttIxtbMzg7NTttIBtbMzg7NTttIAobWzM4OzU7MTA4bS" \
             "gbWzM4OzU7MDcxbS8bWzM4OzU7MDY1bS8bWzM4OzU7MDcxbS8bWzM4OzU7MDcxbS8bWzM4OzU7MDcxbS8bWzM4OzU7MDcxbS8bWzM4" \
             "OzU7MDcxbS8bWzM4OzU7MDcxbS8bWzM4OzU7MDcxbS8bWzM4OzU7MTA3bSgbWzM4OzU7MTQ5bSgbWzM4OzU7MTQ5bSgbWzM4OzU7MT" \
             "Q5bSgbWzM4OzU7MTQ5bSgbWzM4OzU7MTQ5bSgbWzM4OzU7MTQ5bSgbWzM4OzU7MTQ5bSgbWzM4OzU7bSMbWzM4OzU7bSAbWzM4OzU7" \
             "bSAbWzM4OzU7bSMbWzM4OzU7bSMbWzM4OzU7MTczbSgbWzM4OzU7MTczbSgbWzM4OzU7MTczbSgbWzM4OzU7MTczbSgbWzM4OzU7MT" \
             "czbSgbWzM4OzU7MTczbSgbWzM4OzU7MTczbSgbWzM4OzU7MTY3bS8bWzM4OzU7MTY3bS8bWzM4OzU7MTY3bS8bWzM4OzU7MTY3bS8b" \
             "WzM4OzU7MTY3bS8bWzM4OzU7MTY3bS8bWzM4OzU7MTY3bS8bWzM4OzU7MTY3bS8bWzM4OzU7MTY3bS8bWzM4OzU7MTMxbS8KG1szOD" \
             "s1OzEwOG0oG1szODs1OzA2Nm0vG1szODs1OzA2NW0vG1szODs1OzA3MW0vG1szODs1OzA3MW0vG1szODs1OzA3MW0vG1szODs1OzA3" \
             "MW0vG1szODs1OzA3MW0vG1szODs1OzA3MW0vG1szODs1OzA3MW0vG1szODs1OzEwN20oG1szODs1OzE0OW0oG1szODs1OzE0OW0oG1" \
             "szODs1O20jG1szODs1O20oG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1" \
             "O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20sG1szODs1O20lG1szODs1OzE3M20oG1szODs1OzE3M20oG1szOD" \
             "s1OzE3M20oG1szODs1OzE2N20vG1szODs1OzE2N20vG1szODs1OzE2N20vG1szODs1OzE2N20vG1szODs1OzE2N20vG1szODs1OzEz" \
             "MW0vG1szODs1OzEzMW0vG1szODs1OzEzMW0vG1szODs1OzEzMW0vG1szODs1OzEzMW0qChtbMzg7NTttKBtbMzg7NTswNjZtLxtbMz" \
             "g7NTswNjZtLxtbMzg7NTswNjZtLxtbMzg7NTswNzFtLxtbMzg7NTswNzFtLxtbMzg7NTswNzFtLxtbMzg7NTswNzFtLxtbMzg7NTsw" \
             "NzFtLxtbMzg7NTswNzFtLxtbMzg7NTttJhtbMzg7NTttIBtbMzg7NTttIBtbMzg7NTttIBtbMzg7NTttIBtbMzg7NTttIBtbMzg7NT" \
             "ttIBtbMzg7NTttKBtbMzg7NTttIxtbMzg7NTsxNDltKBtbMzg7NTsxNzltIxtbMzg7NTsxNzltIxtbMzg7NTttIxtbMzg7NTttIxtb" \
             "Mzg7NTttIBtbMzg7NTttIBtbMzg7NTttIBtbMzg7NTttIBtbMzg7NTttIBtbMzg7NTttIBtbMzg7NTttKBtbMzg7NTsxMzFtLxtbMz" \
             "g7NTsxMzFtLxtbMzg7NTsxMzFtLxtbMzg7NTsxMzFtLxtbMzg7NTsxMzFtLxtbMzg7NTsxMzFtLxtbMzg7NTsxMzFtKhtbMzg7NTsx" \
             "MzFtKhtbMzg7NTsxMzFtKgobWzM4OzU7bSgbWzM4OzU7MDY2bS8bWzM4OzU7MDY2bS8bWzM4OzU7MDY2bS8bWzM4OzU7MDY2bS8bWz" \
             "M4OzU7MDcybS8bWzM4OzU7MDczbSgbWzM4OzU7MDczbSgbWzM4OzU7MDczbSgbWzM4OzU7MDczbSgbWzM4OzU7bSYbWzM4OzU7bSAb" \
             "WzM4OzU7bSAbWzM4OzU7bSAbWzM4OzU7bSgbWzM4OzU7MTA3bSgbWzM4OzU7MTQ5bSMbWzM4OzU7MTQ5bSgbWzM4OzU7MTQ5bSgbWz" \
             "M4OzU7MTQ5bSgbWzM4OzU7MTc5bSMbWzM4OzU7MTc5bSMbWzM4OzU7MTc5bSMbWzM4OzU7MTc5bSMbWzM4OzU7MTc5bSMbWzM4OzU7" \
             "MTczbSgbWzM4OzU7bSgbWzM4OzU7bSAbWzM4OzU7bSAbWzM4OzU7bSAbWzM4OzU7bSgbWzM4OzU7MTY3bS8bWzM4OzU7MTY3bS8bWz" \
             "M4OzU7MTY3bS8bWzM4OzU7MTY3bS8bWzM4OzU7MTMxbS8bWzM4OzU7MTMxbSobWzM4OzU7MTMxbSobWzM4OzU7MTMxbSobWzM4OzU7" \
             "MTMxbSoKG1szODs1O20oG1szODs1OzA3M20oG1szODs1OzA3M20oG1szODs1OzA3M20oG1szODs1OzA3M20oG1szODs1OzA3M20oG1" \
             "szODs1OzA3M20oG1szODs1OzA3M20oG1szODs1OzA3M20oG1szODs1OzA3M20oG1szODs1O20mG1szODs1O20gG1szODs1O20gG1sz" \
             "ODs1O20gG1szODs1O20oG1szODs1OzAzMm0vG1szODs1OzAzMm0vG1szODs1OzAzMm0vG1szODs1OzAzMm0vG1szODs1OzA3Mm0oG1" \
             "szODs1OzE3OW0jG1szODs1OzE2N20vG1szODs1OzEzMW0qG1szODs1OzEzMW0qG1szODs1OzEzMW0qG1szODs1OzEzMW0qG1szODs1" \
             "O20oG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20oG1szODs1OzE2N20vG1szODs1OzE2N20vG1szODs1OzE2N20vG1" \
             "szODs1OzE2N20vG1szODs1OzE2N20vG1szODs1OzE2N20vG1szODs1OzE2N20vG1szODs1OzE2N20vG1szODs1OzEzMW0vChtbMzg7" \
             "NTttIxtbMzg7NTswMzhtKBtbMzg7NTswMzhtKBtbMzg7NTswNzRtKBtbMzg7NTswNzNtKBtbMzg7NTswNzNtKBtbMzg7NTswNzNtKB" \
             "tbMzg7NTswNzNtKBtbMzg7NTswNzNtKBtbMzg7NTswNzNtKBtbMzg7NTttJhtbMzg7NTttIBtbMzg7NTttIBtbMzg7NTttIBtbMzg7" \
             "NTttKBtbMzg7NTswMzJtLxtbMzg7NTswMzFtLxtbMzg7NTswMjRtKhtbMzg7NTswMjRtKhtbMzg7NTswMjRtKhtbMzg7NTswNjBtKh" \
             "tbMzg7NTswOTVtKhtbMzg7NTswOTVtKhtbMzg7NTswOTVtKhtbMzg7NTsxMzFtKhtbMzg7NTsxMzFtKhtbMzg7NTttKBtbMzg7NTtt" \
             "IBtbMzg7NTttIBtbMzg7NTttIBtbMzg7NTttKBtbMzg7NTsxNjdtLxtbMzg7NTsxNjdtLxtbMzg7NTsxNjdtLxtbMzg7NTsxNjdtLx" \
             "tbMzg7NTsxNjdtLxtbMzg7NTsxNjdtLxtbMzg7NTsxMzFtLxtbMzg7NTsxMzJtLxtbMzg7NTsxMzFtLwobWzM4OzU7bSMbWzM4OzU7" \
             "MDMybS8bWzM4OzU7MDM4bS8bWzM4OzU7MDM4bSgbWzM4OzU7MDM4bSgbWzM4OzU7MDM4bSgbWzM4OzU7MDM4bSgbWzM4OzU7MDc0bS" \
             "gbWzM4OzU7MDczbSgbWzM4OzU7MDczbSgbWzM4OzU7bSYbWzM4OzU7bSAbWzM4OzU7bSAbWzM4OzU7bSAbWzM4OzU7bSAbWzM4OzU7" \
             "bS8bWzM4OzU7MDI0bSobWzM4OzU7MDI0bSobWzM4OzU7MDI0bSobWzM4OzU7MDI0bSobWzM4OzU7MDYwbSobWzM4OzU7MDk1bSobWz" \
             "M4OzU7MDk1bSobWzM4OzU7MDk1bSobWzM4OzU7bS8bWzM4OzU7bSgbWzM4OzU7bSAbWzM4OzU7bSAbWzM4OzU7bSAbWzM4OzU7bSAb" \
             "WzM4OzU7bSgbWzM4OzU7MTY3bS8bWzM4OzU7MTMxbS8bWzM4OzU7MTMybS8bWzM4OzU7MTMybS8bWzM4OzU7MTMybS8bWzM4OzU7MT" \
             "MybS8bWzM4OzU7MTMybS8bWzM4OzU7MTY3bS8bWzM4OzU7MTY3bS8KG1szODs1OzA3NG0jG1szODs1OzAzMm0vG1szODs1OzAzMm0v" \
             "G1szODs1OzAzMm0vG1szODs1OzAzOG0oG1szODs1OzAzOG0oG1szODs1OzAzOG0oG1szODs1OzAzOG0oG1szODs1OzAzMW0vG1szOD" \
             "s1OzAzMW0vG1szODs1OzAyNG0qG1szODs1O20qG1szODs1O20jG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gG1sz" \
             "ODs1O20gG1szODs1O20gG1szODs1O20uG1szODs1O20vG1szODs1O20uG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O2" \
             "0gG1szODs1O20gG1szODs1O20gG1szODs1O20oG1szODs1OzA5Nm0vG1szODs1OzEzMW0vG1szODs1OzEzMW0vG1szODs1OzEzMW0v" \
             "G1szODs1OzEzMm0vG1szODs1OzEzMm0vG1szODs1OzEzMm0vG1szODs1OzE2N20vG1szODs1OzE2N20vG1szODs1OzE2N20vG1szOD" \
             "s1OzE2N20vChtbMzg7NTswNzRtIxtbMzg7NTswMzJtLxtbMzg7NTswMzJtLxtbMzg7NTswMzJtLxtbMzg7NTswNjdtLxtbMzg7NTsw" \
             "NjdtLxtbMzg7NTswMzFtLxtbMzg7NTswMzFtLxtbMzg7NTswMzFtLxtbMzg7NTswMzFtLxtbMzg7NTswMjRtKhtbMzg7NTswMjRtKh" \
             "tbMzg7NTswNjBtKhtbMzg7NTswNjBtKhtbMzg7NTswNjBtKhtbMzg7NTswNjBtKhtbMzg7NTttKBtbMzg7NTttIBtbMzg7NTttIBtb" \
             "Mzg7NTttIBtbMzg7NTttIBtbMzg7NTttIBtbMzg7NTttIBtbMzg7NTttLhtbMzg7NTttKBtbMzg7NTswOTZtLxtbMzg7NTswOTZtLx" \
             "tbMzg7NTswOTZtLxtbMzg7NTsxMzFtLxtbMzg7NTsxMzFtLxtbMzg7NTsxMzFtLxtbMzg7NTsxMzFtLxtbMzg7NTsxMzFtLxtbMzg7" \
             "NTsxMzFtLxtbMzg7NTsxMzFtLxtbMzg7NTsxMzJtLxtbMzg7NTsxMzFtLxtbMzg7NTsxNjdtLxtbMzg7NTsxNjdtLxtbMzg7NTsxNj" \
             "dtLwobWzM4OzU7bSgbWzM4OzU7bSgbWzM4OzU7MDY3bS8bWzM4OzU7MDY3bS8bWzM4OzU7MDY3bS8bWzM4OzU7MDY3bS8bWzM4OzU7" \
             "MDY3bS8bWzM4OzU7MDMxbS8bWzM4OzU7MDMxbS8bWzM4OzU7MDMxbS8bWzM4OzU7MDI0bSobWzM4OzU7MDI0bSobWzM4OzU7MDI0bS" \
             "obWzM4OzU7MDI0bSobWzM4OzU7MDYwbSobWzM4OzU7MDYwbSobWzM4OzU7MDYwbSobWzM4OzU7MDYwbSobWzM4OzU7MDYwbSobWzM4" \
             "OzU7MDYwbSobWzM4OzU7bSobWzM4OzU7MDk2bS8bWzM4OzU7MDk2bS8bWzM4OzU7MDk2bS8bWzM4OzU7MDk2bS8bWzM4OzU7MDk2bS" \
             "8bWzM4OzU7MDk2bS8bWzM4OzU7MTMxbS8bWzM4OzU7MTMxbS8bWzM4OzU7MTMxbS8bWzM4OzU7MTMxbS8bWzM4OzU7MTMxbS8bWzM4" \
             "OzU7MTMxbS8bWzM4OzU7MTMxbS8bWzM4OzU7MTMybS8bWzM4OzU7MTMybS8bWzM4OzU7MTMybS8bWzM4OzU7MTMybS8bWzM4OzU7MT" \
             "MxbS8bWzM4OzU7bSMKG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20uG1szODs1O20jG1szODs1OzA2" \
             "N20vG1szODs1OzA2N20vG1szODs1OzA2N20vG1szODs1OzAzMW0vG1szODs1OzAyNG0qG1szODs1OzAyNG0qG1szODs1OzAyNG0qG1" \
             "szODs1OzAyNG0qG1szODs1OzAyNG0qG1szODs1OzA2MG0qG1szODs1OzA2MG0qG1szODs1OzA2MG0qG1szODs1OzA2MG0qG1szODs1" \
             "OzA2MG0qG1szODs1OzA2MG0qG1szODs1OzIzOW0qG1szODs1OzIzOW0qG1szODs1OzIzOW0qG1szODs1OzIzOW0qG1szODs1OzA5NW" \
             "0vG1szODs1OzEzMW0vG1szODs1OzEzMW0vG1szODs1OzEzMW0vG1szODs1OzEzMW0vG1szODs1OzEzMW0vG1szODs1OzEzMW0vG1sz" \
             "ODs1OzEzMm0vG1szODs1OzEzMm0vG1szODs1OzEzMm0vG1szODs1O20lG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O2" \
             "0gChtbMzg7NTttIBtbMzg7NTttIBtbMzg7NTttIBtbMzg7NTttIBtbMzg7NTttIBtbMzg7NTttIBtbMzg7NTttIBtbMzg7NTttIBtb" \
             "Mzg7NTttIBtbMzg7NTttJhtbMzg7NTswMjRtKhtbMzg7NTswMjRtKhtbMzg7NTswMjRtKhtbMzg7NTswMjRtKhtbMzg7NTswMjRtKh" \
             "tbMzg7NTswMjRtKhtbMzg7NTswMjRtKhtbMzg7NTswNjBtKhtbMzg7NTswNjBtKhtbMzg7NTswNjBtKhtbMzg7NTswNjBtKhtbMzg7" \
             "NTsyMzltKhtbMzg7NTsyMzltKhtbMzg7NTsyMzltKhtbMzg7NTswOTVtKhtbMzg7NTswOTVtKhtbMzg7NTswOTVtKhtbMzg7NTswOT" \
             "VtKhtbMzg7NTswOTVtKhtbMzg7NTswOTVtKhtbMzg7NTsxMzFtLxtbMzg7NTttQBtbMzg7NTttIBtbMzg7NTttIBtbMzg7NTttIBtb" \
             "Mzg7NTttIBtbMzg7NTttIBtbMzg7NTttIBtbMzg7NTttIBtbMzg7NTttIAobWzM4OzU7bSAbWzM4OzU7bSAbWzM4OzU7bSAbWzM4Oz" \
             "U7bSAbWzM4OzU7bSAbWzM4OzU7bSAbWzM4OzU7bSAbWzM4OzU7bSAbWzM4OzU7bSAbWzM4OzU7bSAbWzM4OzU7bSAbWzM4OzU7bSAb" \
             "WzM4OzU7bSAbWzM4OzU7bSgbWzM4OzU7bSobWzM4OzU7MDI0bSobWzM4OzU7MDI0bSobWzM4OzU7MDI0bSobWzM4OzU7MDYwbSobWz" \
             "M4OzU7MDYwbSobWzM4OzU7MDYwbSobWzM4OzU7MjM5bSobWzM4OzU7MjQwbSobWzM4OzU7MDk1bSobWzM4OzU7MDk1bSobWzM4OzU7" \
             "MDk1bSobWzM4OzU7bS8bWzM4OzU7bSAbWzM4OzU7bSAbWzM4OzU7bSAbWzM4OzU7bSAbWzM4OzU7bSAbWzM4OzU7bSAbWzM4OzU7bS" \
             "AbWzM4OzU7bSAbWzM4OzU7bSAbWzM4OzU7bSAbWzM4OzU7bSAbWzM4OzU7bSAbWzM4OzU7bSAKG1szODs1O20gG1szODs1O20gG1sz" \
             "ODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O2" \
             "0gG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20uG1szODs1O20oG1sz" \
             "ODs1OzAyNG0qG1szODs1OzA2MG0qG1szODs1OzA5NW0qG1szODs1O20vG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O2" \
             "0gG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gG1sz" \
             "ODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gG1szODs1O20gChtbMG0="

    print(base64.b64decode(banner).decode("utf-8"))


def print_version() -> None:
    """Prints this package's version number
    """
    print(f'{{cookiecutter.python_name}} version: { {{cookiecutter.python_name}}.core.get_version()}\n')


@click.group()
def cli() -> None:
    """{{cookiecutter.description}}
    """
    print_banner()


@click.command()
def version() -> None:
    """Print version and exit
    """
    print_version()


cli.add_command(version)

if __name__ == '__main__':
    cli()
