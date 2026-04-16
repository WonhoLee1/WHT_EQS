import unicodedata

def get_display_width(s):
    width = 0
    for char in str(s):
        cp = ord(char)
        category = unicodedata.category(char)
        if category in ('Mn', 'Me', 'Cf') or cp == 0xFE0F:
            continue
        status = unicodedata.east_asian_width(char)
        if status in ('W', 'F'):
            width += 2
        elif cp > 0xFFFF:
            width += 2
        elif 0x2000 <= cp <= 0x2BFF:
            if 0x2190 <= cp <= 0x21FF or 0x2500 <= cp <= 0x257F:
                width += 1
            else:
                width += 2
        elif status == 'A':
            width += 1
        else:
            width += 1
    return width

test_strings = [
    "🔄 twist_x",
    "📐 rbe_reaction",
    "📏 u_static",
    "⬆️ lift_br",
    "🌫️ pressure_z",
    "🌍 Global",
    "🎸 modes",
    "🟢 ▼",
    "❌"
]

for s in test_strings:
    print(f"[{s}]: width={get_display_width(s)}")
