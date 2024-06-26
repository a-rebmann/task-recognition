COMPLETE_INDICATORS = ["submit", "save", "ok", "confirm", "apply", "bernehmen", "cancel", "add"]

OVERHEAD_INDICATORS = ["reload", "refresh", "open", "login", "log in", "username", "password", "signin", "sign in",
                       "sign out", "log out", "sign up", "anmeldung"]

COMPLETE_INDICATORS_FULL = ["submit", "save", "ok", "confirm", "apply", "add", "cancel", "close", "delete", "done",
                            "download", "finish", "next", "ok", "post", "reject", "send", "update", "complete", "abort",
                            "upload", "fertig", "speichern", "anwenden", "bernehmen"]

TERMS_FOR_MISSING = ['MISSING', 'UNDEFINED', 'undefined', 'missing', 'none', 'nan', 'empty', 'empties', 'unknown',
                     'other', 'others', 'na', 'nil', 'null', '', "", ' ', '<unknown>', "0;n/a", "NIL", 'undefined',
                     'missing', 'none', 'nan', 'empty', 'empties', 'unknown', 'other', 'others', 'na',
                     'nil', 'null', '', ' ']

# labels
LABEL = "Task"
INDEX = "idx"
OPERATIONS_ID = "operations_id"
PRED_LABEL = "pred_label"
TIMESTAMP = "timeStamp"
APPLICATION = "targetApp"
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

context_attributes = ["target.workbookName", "target.sheetName", "eventType", "target.name", "targetApp",
                      "target.innerText", "target.tagName"]

context_attributes_l = ["eventType", "target.name", "target.id", "target.css", "target.workbookName",
                        "target.sheetName", "target.innerText", "target.tagName", "x", "y"]  # SAP FIORI

semantic_attributes = ["target.innerText", "target.name"]
value_attributes = ["target.innerText", "url", "target.value", "content"]

ui_object_types = {'radio', 'system', 'microsoft edge', 'firefox', 'value', 'sheet', 'app', 'form', 'yahoo', 'web page',
                   'application', 'de', 'span', 'program', 'username', 'option', 'file', 'webpage', 'safari', 'href',
                   'div', 'cell', 'document', 'details', 'button', 'chrome', 'power point', 'picture', 'text', 'check box',
                   'opera', 'response', 'tool', 'mail', 'com', 'browser', 'power pointerp', 'gmail', 'image', 'label',
                   'checkbox', 'outlook', 'link', 'radio button', 'net', 'field', 'site', 'excel', 'url', 'password',
                   'page', 'org', 'textarea', 'server', 'workbook', 'website', 'clipboard', 'line', 'single', 'value',
                   'range'}

COMMON_FILE_EXTENSIONS = [
    "doc",
    "docx",
    "xls",
    "xlsx",
    "ppt",
    "pptx",
    "pdf",
    "txt",
    "csv",
    "jpg",
    "jpeg",
    "png",
    "gif",
    "bmp",
    "tiff",
    "tif",
    "svg",
    "mp3",
    "mp4",
    "avi",
    "mov",
    "wmv",
    "flv",
    "wav",
    "wma",
    "ogg",
    "aac",
    "zip",
    "rar",
    "7z",
    "tar",
    "gz",
    "bz2",
    "xz",
    "iso",
    "img",
    "exe",
    "msi",
    "dmg",
    "pkg",
    "deb",
    "rpm",
    "jar",
    "java",
    "py",
    "c",
    "cpp",
    "h",
    "hpp",
    "cs",
    "vb",
    "vbs",
    "js",
    "ts",
    "css",
    "html",
    "htm",
    "php",
    "xml",
    "json",
    "sql",
    "db",
    "dbf",
    "mdb",
    "accdb",
    "dbx",
    "pst",
    "odt",
    "ods",
    "odp",
    "odg",
    "odf",
    "odb",
    "wps",
    "wpd",
    "eps",
    "ps",
    "prn",
    "xps",
    "dot",
    "dotx",
    "rtf",
    "log",
    "bak",
    "tmp"
]

UI_ACTION_WORDS = [
    "add",
    "apply",
    "approve",
    "back",
    "browse",
    "cancel",
    "clear",
    "close",
    "confirm",
    "copy",
    "create",
    "customize",
    "delete",
    "docs",
    "done",
    "download",
    "drop",
    "edit",
    "empty trash",
    "export",
    "filter",
    "find",
    "finish",
    "follow",
    "get help",
    "hide",
    "import",
    "input",
    "insert",
    "launch",
    "learn more",
    "login"
    "log in",
    "logout",
    "log out",
    "move",
    "move to trash",
    "new",
    "next",
    "ok",
    "paste",
    "play",
    "post",
    "preview",
    "print",
    "redo",
    "refresh",
    "reject",
    "remove",
    "reply",
    "reset",
    "restore",
    "restore all",
    "restore defaults",
    "run",
    "save",
    "save as",
    "search",
    "select",
    "select all",
    "send",
    "show",
    "sign up",
    "sort",
    "start",
    "submit",
    "top",
    "undo",
    "update",
    "upload",
    "view details",
    "click",
    "double click",
    "right click",
    "hover",
    "scroll",
    "drag",
    "drop",
    "select",
    "type",
    "enter",
    "press",
    "wrong"
]
