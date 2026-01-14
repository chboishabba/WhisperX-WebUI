from gradio_i18n import Translate, gettext as _

try:
    AUTOMATIC_DETECTION = _("Automatic Detection")
except LookupError:
    AUTOMATIC_DETECTION = "Automatic Detection"
GRADIO_NONE_STR = ""
GRADIO_NONE_NUMBER_MAX = 9999
GRADIO_NONE_NUMBER_MIN = 0
