class EventManager:
    def fuse(self, track, helmet, weapon, fight):
        events = []

        if helmet and not helmet["helmet"]:
            events.append({
                "type": "helmet_violation",
                "label": "NO HELMET",
                "confidence": helmet["confidence"],
                "details": "Rider without helmet",
                "color": (0, 0, 255)
            })

        if weapon:
            events.append({
                "type": "weapon_detected",
                "label": "WEAPON",
                "confidence": weapon["confidence"],
                "details": "Weapon detected",
                "color": (0, 0, 255)
            })

        return events
