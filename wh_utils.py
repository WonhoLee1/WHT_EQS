import unicodedata

def get_display_width(s):
    """문자열의 실제 콘솔 출력 너비를 시각적으로 계산합니다 (한글 2, 영문 1, 이모지 2)."""
    width = 0
    for char in str(s):
        # East Asian Width (W: Wide, F: Fullwidth)는 2칸을 차지함
        status = unicodedata.east_asian_width(char)
        if status in ('W', 'F'):
            width += 2
        # 이모지 (Surrogate pairs) 및 특정 유니코드 범위 처리
        elif ord(char) > 0xFFFF or ord(char) in [0x231A, 0x231B, 0x23E9, 0x23EA, 0x2600, 0x2705]:
            width += 2
        else:
            width += 1
    return width

# Contextual Emoji Database (ICE Engine)
WH_EMOJI_MAP = {
    # Materials & Properties
    'E': '💎', 'modulus': '💎', 'stiffness': '🧱', 'thickness': '📏', 'base_t': '📏',
    'rho': '☁️', 'density': '☁️', 'mass': '⚖️', 'weight': '⚖️',
    # Physics & Results
    'stress': '🌋', 'strain': '🧶', 'disp': '📏', 'u_static': '📏', 
    'reaction': '📐', 'force': '☄️', 'moment': '🌀', 'torque': '🌀',
    # Modal & Dynamics
    'freq': '🌊', 'hz': '🌊', 'mode': '🎸', 'mac': '🧬', 'eigen': '🎼',
    # Optimization & Logic
    'loss': '📉', 'error': '📉', 'terr': '📉', 'iter': '⏱️', 'target': '🎯', 
    'reg': '🛡️', 'status': '🔔', 'quality': '🏅',
    # Case Types
    'twist': '🔄', 'bend': '🧱', 'lift': '⬆️', 'cantilever': '🏗️', 'pressure': '🌫️',
    'global': '🌍', 'local': '📍'
}

def get_smart_emoji(text):
    """문맥 키워드를 분석하여 적절한 이모지를 반환합니다."""
    text_lower = str(text).lower()
    for key, emoji in WH_EMOJI_MAP.items():
        if key in text_lower:
            return emoji
    return ""

class WHTable:
    """
    [WHTools Premium Table Engine]
    Unicode Box-drawing 문자와 이모지를 활용하여 콘솔에 미려한 테이블을 출력합니다.
    (ICE Engine 탑재: 지능형 이모지 자동 삽입 지원)
    """
    def __init__(self, headers, title=None, emoji_map=None, use_smart_emoji=True):
        self.use_smart_emoji = use_smart_emoji
        # 헤더에도 스마트 이모지 자동 적용
        self.headers = [f"{get_smart_emoji(h)} {h}".strip() if use_smart_emoji else h for h in headers]
        self.title = title
        self.rows = []
        self.emoji_map = emoji_map or {}
        self.aligns = ['left'] * len(headers)

    def set_aligns(self, aligns):
        self.aligns = aligns

    def add_row(self, row, smart_cols=None):
        """
        row: 데이터 리스트
        smart_cols: 스마트 이모지를 적용할 컬럼 인덱스 리스트 (None이면 전체 적용 시도)
        """
        processed_row = []
        for i, item in enumerate(row):
            val = str(item)
            if self.use_smart_emoji:
                if smart_cols is None or i in smart_cols:
                    emoji = get_smart_emoji(val)
                    if emoji and emoji not in val: # 중복 삽입 방지
                        val = f"{emoji} {val}".strip()
            processed_row.append(val)
        self.rows.append(processed_row)

    def _get_widths(self):
        # 헤더별 출력 너비 계산 (한글/영문 보정)
        widths = [get_display_width(h) for h in self.headers]
        for row in self.rows:
            for i, item in enumerate(row):
                widths[i] = max(widths[i], get_display_width(item))
        return widths

    def render(self):
        widths = self._get_widths()
        def pad(s, width, align='left'):
            s = str(s)
            curr_w = get_display_width(s)
            padding = " " * (width - curr_w)
            if align == 'right': return padding + s
            if align == 'center': return " "*(len(padding)//2) + s + " "*(len(padding)-len(padding)//2)
            return s + padding

        out = []
        if self.title:
            out.append(f"\n🚀 {self.title}")
            
        # Top Border
        out.append("┏" + "┳".join(["━"*(w+2) for w in widths]) + "┓")
        # Headers
        out.append("┃ " + " ┃ ".join([pad(h, widths[i], 'center') for i, h in enumerate(self.headers)]) + " ┃")
        # Separator
        out.append("┣" + "╋".join(["━"*(w+2) for w in widths]) + "┫")
        # Rows
        for row in self.rows:
            out.append("┃ " + " ┃ ".join([pad(row[i], widths[i], self.aligns[i]) for i in range(len(row))]) + " ┃")
        # Bottom Border
        out.append("┗" + "┻".join(["━"*(w+2) for w in widths]) + "┛")
        return "\n".join(out)

    def print(self):
        # encoding 설정이 필요한 경우 (Windows 콘솔 환경 대응)
        try:
            rendered = self.render()
            print(rendered)
        except UnicodeEncodeError:
            # 인코딩 에러 시 안전한 문자로 대체하여 출력 시도
            print(self.render().encode('ascii', 'replace').decode('ascii'))

# 향후 추가될 범용 유틸리티들 (파일 입출력, 시스템 체크 등)
def wh_print_banner(text):
    print("\n" + "="*80)
    print(f" {text}".center(80))
    print("="*80 + "\n")
