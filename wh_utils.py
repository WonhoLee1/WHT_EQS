import unicodedata

def get_display_width(s):
    """
    문자열의 실제 콘솔 출력 너비를 시각적으로 계산합니다.
    (Variation Selector 16 대응 및 전각/반각 정교화)
    """
    s_str = str(s)
    width = 0
    i = 0
    while i < len(s_str):
        char = s_str[i]
        cp = ord(char)
        
        # Variation Selector 16 (\uFE0F) 검색 - 앞에 오는 문자를 이모지화(전각) 함
        has_v16 = False
        if i + 1 < len(s_str) and ord(s_str[i+1]) == 0xFE0F:
            has_v16 = True
            
        category = unicodedata.category(char)
        # 조합 문자는 너비 계산에서 제외
        if category in ('Mn', 'Me', 'Cf'):
            i += 1
            continue
        if cp == 0xFE0F:
            i += 1
            continue
            
        status = unicodedata.east_asian_width(char)
        # 전각 문자(W, F), SMP 영역(대부분의 이모지), 또는 V16이 붙은 경우 너비 2
        if status in ('W', 'F') or cp > 0xFFFF or has_v16:
            width += 2
        elif 0x2000 <= cp <= 0x2BFF:
            # 기호 및 화살표 영역 관리
            if 0x2190 <= cp <= 0x21FF or 0x2500 <= cp <= 0x257F: 
                width += 1
            else:
                width += 2
        elif status == 'A':
            # Ambiguous 문자는 터미널/폰트에 따라 다르나, 
            # 윈도우 터미널 표준에 더 가깝게 1로 처리 (V16이 없을 경우)
            width += 1
        else:
            width += 1
        i += 1
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
            s_str = str(s)
            curr_w = get_display_width(s_str)
            total_pad = max(0, width - curr_w)
            
            if align == 'right':
                return " " * total_pad + s_str
            elif align == 'center':
                left_pad = total_pad // 2
                right_pad = total_pad - left_pad
                return " " * left_pad + s_str + " " * right_pad
            else: # left
                return s_str + " " * total_pad

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
