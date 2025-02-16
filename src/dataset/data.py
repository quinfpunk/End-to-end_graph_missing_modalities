class eICUData:

    def __init__(
            self,
            icu_id,
            admission_id,
            patient_id,
            icu_duration,
            hospital_id,
            mortality,
            readmission,
            age,
            gender,
            ethnicity,
    ):
        self.icu_id = icu_id  # str
        self.admission_id = admission_id  # str
        self.patient_id = patient_id  # str
        self.icu_duration = icu_duration  # int
        self.hospital_id = hospital_id  # int
        self.mortality = mortality  # bool, end of icu stay mortality
        self.readmission = readmission  # bool, 15-day icu readmission
        self.age = age  # int
        self.gender = gender  # str
        self.ethnicity = ethnicity  # str

        # list of tuples (timestamp in min (int), type (str), list of codes (str))
        self.diagnosis = []
        self.treatment = []
        self.medication = []

        # (list of types (str), list of codes (str))
        self.trajectory = []

        # labs
        # (timestamp in min (int), list of (item_id, value))
        self.lab = []
        # numpy array
        self.labvectors = None

        # apacheapsvar
        # numpy array
        self.apacheapsvar = None

    def __repr__(self):
        return f"ICU ID-{self.icu_id} ({self.icu_duration} min): " \
               f"mortality-{self.mortality}, " \
               f"readmission-{self.readmission}"

    def data_print(self):
        print(f" ICU ID: {self.icu_id} ({self.icu_duration} min) \n Age: {self.age} \n Gender: {self.gender} \n Ethnicity: {self.ethnicity}")
        print(f"  Diagnosis: {self.diagnosis} \n  Treatment: {self.treatment} \n  Medication: {self.medication} \n  "
              f"Lab: {self.lab} \n "
              f"Labvectors: {self.labvectors}"
              f"\n  Apacheapsvar: {self.apacheapsvar}")
        # print(f"Diagnosis: {len(self.diagnosis)}   Treatment: {len(self.treatment)}   Medication: {len(self.medication)}")
        # print(f"Trajectory[0]: {self.trajectory[0]}")
        # print(f"Trajectory[1]: {self.trajectory[1]}")
        return
