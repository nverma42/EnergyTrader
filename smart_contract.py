from argparse import Action
from pyteal import *

def store_price():
    h = Txn.application_args[1]
    price_key = Concat(Bytes("price_"), h)
    submitted_by_key = Bytes("submitted_by")
    paid_key = Concat(Bytes("paid_"), h)
    return Seq([
        App.globalPut(price_key, Btoi(Txn.application_args[2])),
        App.globalPut(submitted_by_key, Txn.sender()),
        App.globalPut(paid_key, Int(0)),
        Return(Int(1))
    ])

def release_payment():
    h = Txn.application_args[1]
    price_key = Concat(Bytes("price_"), h)
    price = App.globalGet(price_key)

    submitted_by_key = Bytes("submitted_by")
    submitted_by = App.globalGet(submitted_by_key)

    paid_key = Concat(Bytes("paid_"), h)
    paid = App.globalGet(paid_key)

    seller_address = Txn.accounts[1]  # Assuming the seller's address is the second account in the transaction
    amount = Btoi(Txn.application_args[3])

    return Seq([
        Assert(Txn.sender() == submitted_by),
        Assert(price > Int(0)),
        Assert(paid == Int(0)),
        InnerTxnBuilder.Begin(),
        InnerTxnBuilder.SetFields({
            TxnField.type_enum: TxnType.Payment,
            TxnField.receiver: seller_address,
            TxnField.amount: amount,
        }),
        InnerTxnBuilder.Submit(),
        Return(Int(1))
    ])

def approval_program():
    action = Txn.application_args[0]
    return Cond(
        [Txn.application_id() == Int(0), Approve()],  # on create
        [Txn.on_completion() == OnComplete.DeleteApplication, Return(Int(1))],
        [Txn.on_completion() == OnComplete.UpdateApplication, Return(Int(1))],
        [Txn.application_args[0] == Bytes("store_price"), store_price()],
        [Txn.application_args[0] == Bytes("release_payment"), release_payment()],
        [Int(1), Reject()])

def clear_state_program():
    return Approve()


