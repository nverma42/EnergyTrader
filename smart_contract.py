from pyteal import *

def approval_program():
    return Cond(
        [Txn.application_id() == Int(0), Approve()],  # on create
        [Txn.on_completion() == OnComplete.NoOp, 
         App.globalPut(Bytes("settlement_price"), Btoi(Txn.application_args[0])), Approve()]
    )

def clear_state_program():
    return Approve()


