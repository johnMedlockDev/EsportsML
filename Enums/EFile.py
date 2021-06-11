from enum import Enum, unique


@unique
class EFile(Enum):
    PLAYER = "Player"
    TEAM = "Team"

    def describe(self):
        return self.name, self.value
