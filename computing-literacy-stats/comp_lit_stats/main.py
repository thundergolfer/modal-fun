import modal

modal_workspace_username = "thundergolfer"
app_name = "comp-lit-stats"
volume = modal.SharedVolume().persist(f"{app_name}-vol")

stub = modal.Stub(name=app_name)

# TODO: endpoint to accept new stat line

# TODO: endpoint to return summary of community feedback
