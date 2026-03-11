class EchoFeelingEngine:

    def __init__(self):
        print("Echo Feeling Engine Loaded")

    def analyze(self, text=None, emoji=None, sticker=None):

        result = {
            "sentiment": "neutral",
            "confidence": 0.90
        }

        return result