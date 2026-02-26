import json
import numpy as np


class Template:
    def __init__(self, template_dict):
        """Object containing information about a template

        Parameters
        ----------
        template_dict : `dict`
            Python Dictionary of selected template
        """
        self.mass1 = template_dict["mass1"]
        self.mass2 = template_dict["mass2"]
        self.distance = template_dict["distance"]
        self.hp = np.array(template_dict["values_plus"])
        self.hc = np.array(template_dict["values_cross"])
        self.times = np.array(template_dict["times"])


class Templates:
    def __init__(self, loc):
        """Fetches templates for matched filtering from JSON file

        Parameters
        ----------
        loc : `str`
            path of the JSON to import
        """

        self.templateDict = {}
        with open(loc) as f:
            self.templateDict = json.load(f)

        self.template_count = self.templateDict["template_count"]
        self.min_mass = self.templateDict["min_mass"]
        self.max_mass = self.templateDict["max_mass"]
        self.mass_step = self.templateDict["mass_step"]
        self.distances = self.templateDict["distances"]

    def get_template(self, event_index=0, mass1=-1, mass2=-1):
        """Returns a template

        Parameters
        ----------
        event_index : `int`
            index of event stored in uploaded JSON file
        mass1 : `int`
            mass of larger BH in solar masses
        mass2 : `int`
            mass of smaller BH in solar masses

        Returns
        ----------
        Template : `obj`
            Template object

        NOTE
        ----------
        Only change mass1 and mass2 if Templates.mass_step == 1
        """
        if event_index < 0 or event_index >= self.template_count:
            raise Exception("event_index out of bounds")
        if (
            (
                mass1 > self.max_mass
                or mass2 > self.max_mass
                or mass1 < self.min_mass
                or mass2 < self.min_mass
                or mass1 < mass2
            )
            and mass1 != -1
        ):
            raise Exception("mass1 or mass2 out of bounds")

        masses = mass1 - self.min_mass
        jump1 = (masses * (masses+1)) // 2
        jump2 = jump1 + mass2 - self.min_mass

        index = jump2 if mass1 != -1 else event_index

        self.selectedTemplate = self.templateDict["templates"][index]
        return Template(self.selectedTemplate)
