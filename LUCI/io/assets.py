"""Locating the data and model files LUCI ships with."""


def check_luci_path(Luci_path):
    """
    Functionality to check that the user has included the trailing "/" to Luci_path.
    If they have not, we add it.
    """
    if not Luci_path.endswith("/"):
        Luci_path += "/"
        print("We have added a trailing '/' to your Luci_path variable.\n")
        print("Please add this in the future.\n")
    return Luci_path
