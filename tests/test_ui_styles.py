import unittest

from ui.styles import dashboard_css


class UiStylesTests(unittest.TestCase):
    def test_dashboard_css_uses_dark_operational_palette(self):
        css = dashboard_css()

        self.assertIn("#070b10", css)
        self.assertIn("#111827", css)
        self.assertIn("#22d3ee", css)
        self.assertIn(".kairos-shell", css)
        self.assertIn(".status-badge", css)

    def test_dashboard_css_styles_streamlit_cards_and_inputs(self):
        css = dashboard_css()

        self.assertIn('div[data-testid="stVerticalBlockBorderWrapper"]', css)
        self.assertIn('div[data-testid="stFileUploader"]', css)
        self.assertIn('div[data-testid="stMetric"]', css)
        self.assertIn("border-radius: 8px", css)


if __name__ == "__main__":
    unittest.main()
