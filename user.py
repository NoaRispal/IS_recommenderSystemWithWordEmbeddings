class User:
    """
    https://docs.google.com/document/d/1RwI_5aa5GOH2U1Dngx3EeGzeiFFayejrjy5YLjh4RB8/edit?tab=t.0#heading=h.gfqkvyvr6il2
    """
    level_value = dict(zip(["highschool","undergraduate","graduate","phd"], [0,1,2,3]))

    def __init__(self,id,fullname,hourly_rate,subject,level,preferred_learning_mode,special_needs,location,email):
        self.id = id
        self.fullname=fullname
        self.hourly_rate=hourly_rate
        self.subject=subject
        self.level=level.lower()
        self.preferred_learning_mode = preferred_learning_mode
        self.special_needs = special_needs
        self.location=location
        self.email = email

    def apply_filter(self,filter):
        return [getattr(self, feature) for feature in filter]
    
    @classmethod
    def compare_level(cls,self,other):
        return cls.level_value[self.level] >= cls.level_value[other.level]
    
    @classmethod
    def get_level_distance(cls,self,other):
        return abs(cls.level_value[self.level] - cls.level_value[other.level])


class Tutoree(User):
        def __init__(self,id,fullname, hourly_rate,subject,level,preferred_learning_mode,special_needs,location,email,query):
            super().__init__(id,fullname,hourly_rate,subject,level,preferred_learning_mode,special_needs,location,email)
            self.query=query

class Tutor(User):
    mandatory_features = ["subject","preferred_learning_mode","location"]
    bonus_features = ["hourly_rate","special_needs","level"] 

    def __init__(self,id,fullname, hourly_rate,subject,level,preferred_learning_mode,special_needs,location,email,bio,precise_domain):
        super().__init__(id,fullname,hourly_rate,subject,level,preferred_learning_mode,special_needs,location,email)
        self.bio=bio
        self.precise_domain = precise_domain

    @classmethod
    def get_scoring_bonus_features(cls):
        return cls.mandatory_features,cls.bonus_features

    @staticmethod
    def get_tutor(df, indices):
        subset = df.iloc[indices]
        return [Tutor(**row.to_dict()) for _, row in subset.iterrows()]