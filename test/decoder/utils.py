from spenderq.spenderq import SpenderQ


def open_model():
    sq = SpenderQ("qso.dr1.hiz")
    model = sq.models[-1][0]     # the .i3 network

    print(type(model))
    print(model)  
    return model

