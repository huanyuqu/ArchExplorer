class EnvError(ValueError):
    def __init__(self, code, message, **details):
        super().__init__(message)
        self.code, self.details = code, details

    def to_dict(self):
        return {"code": self.code, "message": str(self), "details": self.details}
