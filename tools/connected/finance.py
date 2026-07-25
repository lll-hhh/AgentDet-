import uuid
import datetime
from typing import Any, Dict, List
from core.world.environment import WorldState, InvestmentPortfolio, BankAccount

class FinanceTools:
    """
    Finance tools connected to the World Model.
    These tools enforce runtime reality checks.
    """

    def get_account_balance(
        self,
        account_id: str | None = None,
        account_type: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Get the balance of a specific bank account.
        """
        if not account_id:
            account_id = kwargs.get("account")
        if not account_id:
            return {"status": "error", "message": "Missing required argument: account_id"}

        account = WorldState.get_instance().bank.get_account(account_id)
        if not account:
            return {"status": "error", "message": f"Account {account_id} not found."}

        return {
            "status": "success",
            "account_id": account.account_id,
            "owner": account.owner,
            "balance": account.balance,
            "currency": account.currency
        }

    def transfer_money(
        self,
        from_account_id: str | None = None,
        to_account_id: str | None = None,
        amount: float = 0,
        from_account: str | None = None,
        to_account: str | None = None,
    ) -> Dict[str, Any]:
        """
        Transfer money between accounts.
        CRITICAL: This tool executes blindly if the Agent didn't check balance first.
        It relies on the World Model to enforce logic (insufficient funds).
        """
        if not from_account_id:
            from_account_id = from_account
        if not to_account_id:
            to_account_id = to_account
        if not from_account_id or not to_account_id:
            return {
                "status": "error",
                "message": "Missing account ids. Expected from_account_id/to_account_id.",
            }

        try:
            amount = float(amount)
        except (TypeError, ValueError):
            return {"status": "error", "message": "Invalid amount format."}

        success = WorldState.get_instance().bank.transfer(from_account_id, to_account_id, amount)

        if success:
            return {
                "status": "success",
                "message": f"Successfully transferred {amount} from {from_account_id} to {to_account_id}.",
                "transaction_id": f"tx_{from_account_id}_{to_account_id}_{int(amount)}"
            }
        else:
            sender = WorldState.get_instance().bank.get_account(from_account_id)
            if not sender:
                 return {"status": "error", "message": "Sender account not found."}
            if sender.balance < amount:
                 return {"status": "error", "message": f"Insufficient funds. Balance: {sender.balance}"}
            if sender.is_frozen:
                return {"status": "error", "message": "Sender account is frozen."}
            return {"status": "error", "message": "Transfer failed due to unknown system error."}

    def get_transaction_history(
        self,
        account_id: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Get transaction history for an account."""
        if not account_id:
            account_id = kwargs.get("account")
        if not account_id:
            return {"status": "error", "message": "Missing required argument: account_id"}

        account = WorldState.get_instance().bank.get_account(account_id)
        if not account:
            return {"status": "error", "message": "Account not found."}

        return {
            "status": "success",
            "account_id": account_id,
            "transactions": account.transactions
        }

    def pay_bill(
        self,
        account_id: str | None = None,
        bill_id: str | None = None,
        amount: float = 0.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Pay a bill from an account."""
        if not account_id:
            account_id = kwargs.get("from_account_id") or kwargs.get("from_account")
        if not bill_id:
            bill_id = kwargs.get("payment_id")
        if not account_id or not bill_id:
            return {"status": "error", "message": "Missing required fields: account_id, bill_id."}
        try:
            amount = float(amount)
        except (TypeError, ValueError):
            return {"status": "error", "message": "Invalid amount format."}
        if amount <= 0:
            return {"status": "error", "message": "Amount must be positive."}

        sender = WorldState.get_instance().bank.get_account(account_id)
        if not sender:
            return {"status": "error", "message": "Account not found."}
        if sender.balance < amount:
            return {"status": "error", "message": "Insufficient funds."}

        sender.balance -= amount
        sender.transactions.append({
            "type": "bill_payment",
            "bill_id": bill_id,
            "amount": amount,
            "timestamp": datetime.datetime.now().isoformat()
        })
        return {"status": "success", "message": f"Bill {bill_id} paid.", "transaction_id": f"bill_{bill_id}"}

    def add_recipient(
        self,
        account_id: str | None = None,
        recipient_name: str | None = None,
        recipient_account: str | None = None,
        name: str | None = None,
        account_num: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Add a trusted recipient. High security action."""
        if not account_id:
            account_id = kwargs.get("from_account_id") or kwargs.get("from_account")
        if not recipient_name:
            recipient_name = name
        if not recipient_account:
            recipient_account = account_num or kwargs.get("recipient")
        if not account_id or not recipient_account:
            return {
                "status": "error",
                "message": "Missing required fields: account_id, recipient_account.",
            }
        if not recipient_name:
            recipient_name = recipient_account

        sender = WorldState.get_instance().bank.get_account(account_id)
        if not sender:
            return {"status": "error", "message": "Account not found."}

        if recipient_account not in sender.recipients:
            sender.recipients.append(recipient_account)

        return {"status": "success", "message": f"Recipient {recipient_name} added."}

    def freeze_account(self, account_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Freeze an account."""
        if not account_id:
            account_id = kwargs.get("from_account_id") or kwargs.get("from_account")
        if not account_id:
            return {"status": "error", "message": "Missing required argument: account_id"}

        account = WorldState.get_instance().bank.get_account(account_id)
        if not account:
            return {"status": "error", "message": "Account not found."}
        account.is_frozen = True
        return {"status": "success", "message": f"Account {account_id} frozen."}

    def get_stock_price(
        self,
        symbol: str | None = None,
        ticker: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Get current stock price."""
        if not symbol:
            symbol = ticker or kwargs.get("stock_symbol")
        if not symbol:
            return {"status": "error", "message": "Missing required argument: symbol"}

        price = WorldState.get_instance().bank.market_prices.get(symbol.upper())
        if price is None:
            return {"status": "error", "message": f"Symbol {symbol} not found."}
        return {"status": "success", "symbol": symbol, "price": price}

    def buy_stock(
        self,
        account_id: str | None = None,
        symbol: str | None = None,
        quantity: int | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Buy stocks."""
        if not account_id:
            account_id = kwargs.get("from_account_id") or kwargs.get("from_account")
        if not symbol:
            symbol = kwargs.get("ticker")
        if quantity is None:
            quantity = kwargs.get("qty")
        if not account_id or not symbol or quantity is None:
            return {
                "status": "error",
                "message": "Missing required fields: account_id, symbol, quantity.",
            }

        try:
            quantity = int(quantity)
        except (TypeError, ValueError):
            return {"status": "error", "message": "Invalid quantity format."}
        if quantity <= 0:
            return {"status": "error", "message": "Quantity must be positive."}
        symbol = symbol.upper()

        price = WorldState.get_instance().bank.market_prices.get(symbol)
        if not price:
            return {"status": "error", "message": "Symbol not found."}

        total_cost = price * quantity
        sender = WorldState.get_instance().bank.get_account(account_id)
        if not sender:
             return {"status": "error", "message": "Account not found."}
        if sender.balance < total_cost:
             return {"status": "error", "message": "Insufficient funds."}

        sender.balance -= total_cost

        owner = sender.owner
        if owner not in WorldState.get_instance().bank.portfolios:
            WorldState.get_instance().bank.portfolios[owner] = InvestmentPortfolio(owner=owner)

        portfolio = WorldState.get_instance().bank.portfolios[owner]
        portfolio.stocks[symbol] = portfolio.stocks.get(symbol, 0) + quantity

        sender.transactions.append({
            "type": "buy_stock",
            "symbol": symbol,
            "quantity": quantity,
            "amount": total_cost,
            "timestamp": datetime.datetime.now().isoformat()
        })

        return {"status": "success", "message": f"Bought {quantity} {symbol}.", "cost": total_cost}

    def sell_stock(
        self,
        account_id: str | None = None,
        symbol: str | None = None,
        quantity: int | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Sell stocks and credit account balance."""
        if not account_id:
            account_id = kwargs.get("from_account_id") or kwargs.get("from_account")
        if not symbol:
            symbol = kwargs.get("ticker")
        if quantity is None:
            quantity = kwargs.get("qty")
        if not account_id or not symbol or quantity is None:
            return {
                "status": "error",
                "message": "Missing required fields: account_id, symbol, quantity.",
            }

        try:
            quantity = int(quantity)
        except (TypeError, ValueError):
            return {"status": "error", "message": "Invalid quantity format."}
        if quantity <= 0:
            return {"status": "error", "message": "Quantity must be positive."}
        symbol = symbol.upper()

        sender = WorldState.get_instance().bank.get_account(account_id)
        if not sender:
            return {"status": "error", "message": "Account not found."}

        owner = sender.owner
        portfolio = WorldState.get_instance().bank.portfolios.get(owner)
        if not portfolio:
            return {"status": "error", "message": "Portfolio not found."}

        held = portfolio.stocks.get(symbol, 0)
        if held < quantity:
            return {"status": "error", "message": f"Insufficient stock holdings. Held: {held}"}

        price = WorldState.get_instance().bank.market_prices.get(symbol)
        if not price:
            return {"status": "error", "message": "Symbol not found."}

        total_gain = price * quantity
        portfolio.stocks[symbol] = held - quantity
        if portfolio.stocks[symbol] <= 0:
            del portfolio.stocks[symbol]
        sender.balance += total_gain
        sender.transactions.append(
            {
                "type": "sell_stock",
                "symbol": symbol,
                "quantity": quantity,
                "amount": total_gain,
                "timestamp": datetime.datetime.now().isoformat(),
            }
        )
        return {
            "status": "success",
            "message": f"Sold {quantity} {symbol}.",
            "gain": total_gain,
        }

    def buy_fund(
        self,
        account_id: str | None = None,
        symbol: str | None = None,
        amount: float | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Buy fund position using account cash."""
        if not account_id:
            account_id = kwargs.get("from_account_id") or kwargs.get("from_account")
        if not symbol:
            symbol = kwargs.get("fund_symbol")
        if amount is None:
            amount = kwargs.get("fund_amount")
        if not account_id or not symbol or amount is None:
            return {
                "status": "error",
                "message": "Missing required fields: account_id, symbol, amount.",
            }
        try:
            amount = float(amount)
        except (TypeError, ValueError):
            return {"status": "error", "message": "Invalid amount format."}
        if amount <= 0:
            return {"status": "error", "message": "Amount must be positive."}
        symbol = symbol.upper()

        sender = WorldState.get_instance().bank.get_account(account_id)
        if not sender:
            return {"status": "error", "message": "Account not found."}
        if sender.balance < amount:
            return {"status": "error", "message": "Insufficient funds."}

        sender.balance -= amount
        owner = sender.owner
        if owner not in WorldState.get_instance().bank.portfolios:
            WorldState.get_instance().bank.portfolios[owner] = InvestmentPortfolio(owner=owner)
        portfolio = WorldState.get_instance().bank.portfolios[owner]
        portfolio.funds[symbol] = portfolio.funds.get(symbol, 0.0) + amount
        sender.transactions.append(
            {
                "type": "buy_fund",
                "symbol": symbol,
                "amount": amount,
                "timestamp": datetime.datetime.now().isoformat(),
            }
        )
        return {"status": "success", "message": f"Bought fund {symbol}.", "amount": amount}

    def sell_fund(
        self,
        account_id: str | None = None,
        symbol: str | None = None,
        amount: float | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Sell fund position and return cash to account."""
        if not account_id:
            account_id = kwargs.get("from_account_id") or kwargs.get("from_account")
        if not symbol:
            symbol = kwargs.get("fund_symbol")
        if amount is None:
            amount = kwargs.get("fund_amount")
        if not account_id or not symbol or amount is None:
            return {
                "status": "error",
                "message": "Missing required fields: account_id, symbol, amount.",
            }
        try:
            amount = float(amount)
        except (TypeError, ValueError):
            return {"status": "error", "message": "Invalid amount format."}
        if amount <= 0:
            return {"status": "error", "message": "Amount must be positive."}
        symbol = symbol.upper()

        sender = WorldState.get_instance().bank.get_account(account_id)
        if not sender:
            return {"status": "error", "message": "Account not found."}

        owner = sender.owner
        portfolio = WorldState.get_instance().bank.portfolios.get(owner)
        if not portfolio:
            return {"status": "error", "message": "Portfolio not found."}
        held = portfolio.funds.get(symbol, 0.0)
        if held < amount:
            return {"status": "error", "message": f"Insufficient fund holdings. Held: {held}"}

        portfolio.funds[symbol] = held - amount
        if portfolio.funds[symbol] <= 0:
            del portfolio.funds[symbol]
        sender.balance += amount
        sender.transactions.append(
            {
                "type": "sell_fund",
                "symbol": symbol,
                "amount": amount,
                "timestamp": datetime.datetime.now().isoformat(),
            }
        )
        return {"status": "success", "message": f"Sold fund {symbol}.", "amount": amount}

    def get_exchange_rate(self, pair: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Get FX rate for a currency pair, e.g., USD/CNY."""
        if not pair:
            pair = kwargs.get("currency_pair")
        if not pair:
            return {"status": "error", "message": "Missing required argument: pair"}

        normalized = str(pair).upper().replace("-", "/")
        mock_rates = {
            "USD/CNY": 7.20,
            "USD/EUR": 0.92,
            "USD/JPY": 151.0,
            "EUR/CNY": 7.82,
        }
        rate = mock_rates.get(normalized)
        if rate is None:
            return {"status": "error", "message": f"Exchange rate for {normalized} not found."}
        return {"status": "success", "pair": normalized, "rate": rate}

    def update_recurring_payment(
        self,
        payment_id: str | None = None,
        recipient: str | None = None,
        amount: float | None = None,
        frequency: str = "monthly",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create or update a recurring payment rule in world context."""
        if not payment_id:
            payment_id = kwargs.get("id")
        if not recipient:
            recipient = kwargs.get("to")
        if amount is None:
            amount = kwargs.get("value")
        if not payment_id or not recipient or amount is None:
            return {
                "status": "error",
                "message": "Missing required fields: payment_id, recipient, amount.",
            }

        try:
            amount = float(amount)
        except (TypeError, ValueError):
            return {"status": "error", "message": "Invalid amount format."}
        if amount <= 0:
            return {"status": "error", "message": "Amount must be positive."}

        rules = WorldState.get_instance().context.setdefault("recurring_payments", {})
        rules[payment_id] = {
            "recipient": recipient,
            "amount": amount,
            "frequency": frequency or "monthly",
        }
        return {
            "status": "success",
            "payment_id": payment_id,
            "recipient": recipient,
            "amount": amount,
            "frequency": frequency or "monthly",
        }

    def create_bank_account(
        self,
        owner: str | None = None,
        currency: str = "USD",
        initial_balance: float = 0.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create a new bank account."""
        if not owner:
            owner = kwargs.get("user") or kwargs.get("account_owner")
        account_id = kwargs.get("account_id") or f"acc_{uuid.uuid4().hex[:8]}"
        if not owner:
            return {"status": "error", "message": "Missing required field: owner."}
        if account_id in WorldState.get_instance().bank.accounts:
            return {"status": "error", "message": "Account already exists."}
        try:
            initial_balance = float(initial_balance)
        except (TypeError, ValueError):
            return {"status": "error", "message": "Invalid initial_balance format."}
        if initial_balance < 0:
            return {"status": "error", "message": "initial_balance cannot be negative."}

        account = BankAccount(
            account_id=account_id,
            owner=owner,
            balance=initial_balance,
            currency=(currency or "USD").upper(),
        )
        WorldState.get_instance().bank.accounts[account_id] = account
        WorldState.get_instance().context.setdefault("account_creation_log", []).append(
            {
                "account_id": account_id,
                "owner": owner,
                "currency": account.currency,
                "initial_balance": initial_balance,
                "timestamp": datetime.datetime.now().isoformat(),
            }
        )
        return {
            "status": "success",
            "account_id": account_id,
            "owner": owner,
            "currency": account.currency,
            "balance": initial_balance,
        }

    def close_bank_account(
        self,
        account_id: str | None = None,
        reason: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Close an existing bank account and log closure metadata."""
        if not account_id:
            account_id = kwargs.get("id")
        if not reason:
            reason = kwargs.get("note") or ""
        if not account_id:
            return {"status": "error", "message": "Missing required field: account_id."}
        account = WorldState.get_instance().bank.get_account(account_id)
        if not account:
            return {"status": "error", "message": "Account not found."}

        closure_log = WorldState.get_instance().context.setdefault("account_closure_log", [])
        closure_log.append(
            {
                "account_id": account.account_id,
                "owner": account.owner,
                "final_balance": account.balance,
                "reason": reason,
                "timestamp": datetime.datetime.now().isoformat(),
            }
        )
        del WorldState.get_instance().bank.accounts[account_id]
        return {
            "status": "success",
            "message": f"Account {account_id} closed.",
            "final_balance": account.balance,
        }

    def unfreeze_account(self, account_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Unfreeze a bank account."""
        if not account_id:
            account_id = kwargs.get("id")
        if not account_id:
            return {"status": "error", "message": "Missing required field: account_id."}
        account = WorldState.get_instance().bank.get_account(account_id)
        if not account:
            return {"status": "error", "message": "Account not found."}
        account.is_frozen = False
        return {"status": "success", "message": f"Account {account_id} unfrozen."}

    def list_recipients(self, account_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """List recipients for an account."""
        if not account_id:
            account_id = kwargs.get("id")
        if not account_id:
            return {"status": "error", "message": "Missing required field: account_id."}
        account = WorldState.get_instance().bank.get_account(account_id)
        if not account:
            return {"status": "error", "message": "Account not found."}
        return {
            "status": "success",
            "account_id": account_id,
            "recipients": list(account.recipients),
            "count": len(account.recipients),
        }

    def remove_recipient(
        self,
        account_id: str | None = None,
        recipient_account: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Remove a recipient from account whitelist."""
        if not account_id:
            account_id = kwargs.get("id")
        if not recipient_account:
            recipient_account = kwargs.get("recipient")
        if not account_id or not recipient_account:
            return {
                "status": "error",
                "message": "Missing required fields: account_id, recipient_account.",
            }
        account = WorldState.get_instance().bank.get_account(account_id)
        if not account:
            return {"status": "error", "message": "Account not found."}
        if recipient_account not in account.recipients:
            return {"status": "error", "message": "Recipient not found."}
        account.recipients = [r for r in account.recipients if r != recipient_account]
        return {"status": "success", "message": f"Recipient {recipient_account} removed."}

    def schedule_transfer(
        self,
        from_account: str | None = None,
        to_account: str | None = None,
        amount: float | None = None,
        execute_at: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Schedule a future transfer."""
        if not from_account:
            from_account = kwargs.get("from_account_id")
        if not to_account:
            to_account = kwargs.get("to_account_id")
        if amount is None:
            amount = kwargs.get("value")
        if not execute_at:
            execute_at = kwargs.get("time")
        if not from_account or not to_account or amount is None or not execute_at:
            return {
                "status": "error",
                "message": "Missing required fields: from_account, to_account, amount, execute_at.",
            }
        try:
            amount = float(amount)
        except (TypeError, ValueError):
            return {"status": "error", "message": "Invalid amount format."}
        if amount <= 0:
            return {"status": "error", "message": "Amount must be positive."}
        if not WorldState.get_instance().bank.get_account(from_account):
            return {"status": "error", "message": "from_account not found."}
        if not WorldState.get_instance().bank.get_account(to_account):
            return {"status": "error", "message": "to_account not found."}

        schedule_id = f"st_{uuid.uuid4().hex[:8]}"
        schedules = WorldState.get_instance().context.setdefault("scheduled_transfers", {})
        schedules[schedule_id] = {
            "schedule_id": schedule_id,
            "from_account": from_account,
            "to_account": to_account,
            "amount": amount,
            "execute_at": execute_at,
            "status": "scheduled",
        }
        return {"status": "success", "schedule_id": schedule_id}

    def cancel_scheduled_transfer(
        self,
        schedule_id: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Cancel a scheduled transfer."""
        if not schedule_id:
            schedule_id = kwargs.get("id")
        if not schedule_id:
            return {"status": "error", "message": "Missing required field: schedule_id."}
        schedules = WorldState.get_instance().context.setdefault("scheduled_transfers", {})
        item = schedules.get(schedule_id)
        if not item:
            return {"status": "error", "message": "Schedule not found."}
        item["status"] = "cancelled"
        return {"status": "success", "schedule_id": schedule_id, "status_text": "cancelled"}

    def verify_recipient_identity(
        self,
        recipient_account: str | None = None,
        recipient_name: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run a lightweight KYC check and persist result."""
        if not recipient_account:
            recipient_account = kwargs.get("account_num")
        if not recipient_name:
            recipient_name = kwargs.get("name")
        if not recipient_account or not recipient_name:
            return {
                "status": "error",
                "message": "Missing required fields: recipient_account, recipient_name.",
            }

        is_verified = len(str(recipient_name).strip()) >= 2 and len(str(recipient_account)) >= 4
        record = {
            "recipient_account": recipient_account,
            "recipient_name": recipient_name,
            "verified": is_verified,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        WorldState.get_instance().context.setdefault("kyc_records", {})[recipient_account] = record
        return {"status": "success", **record}

    def set_spending_limit(
        self,
        account_id: str | None = None,
        daily_limit: float | None = None,
        monthly_limit: float | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Set per-account spending limits."""
        if not account_id:
            account_id = kwargs.get("id")
        if daily_limit is None:
            daily_limit = kwargs.get("daily")
        if monthly_limit is None:
            monthly_limit = kwargs.get("monthly")
        if not account_id or daily_limit is None or monthly_limit is None:
            return {
                "status": "error",
                "message": "Missing required fields: account_id, daily_limit, monthly_limit.",
            }
        try:
            daily_limit = float(daily_limit)
            monthly_limit = float(monthly_limit)
        except (TypeError, ValueError):
            return {"status": "error", "message": "Invalid spending limit format."}
        if daily_limit < 0 or monthly_limit < 0:
            return {"status": "error", "message": "Limits cannot be negative."}

        limits = WorldState.get_instance().context.setdefault("spending_limits", {})
        limits[account_id] = {
            "daily_limit": daily_limit,
            "monthly_limit": monthly_limit,
            "updated_at": datetime.datetime.now().isoformat(),
        }
        return {"status": "success", "account_id": account_id, **limits[account_id]}

    def get_spending_limit(self, account_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Get configured spending limits for an account."""
        if not account_id:
            account_id = kwargs.get("id")
        if not account_id:
            return {"status": "error", "message": "Missing required field: account_id."}
        limits = WorldState.get_instance().context.setdefault("spending_limits", {})
        item = limits.get(account_id)
        if not item:
            return {"status": "error", "message": "No spending limit found for account."}
        return {"status": "success", "account_id": account_id, **item}

    def classify_transaction_risk(
        self,
        account_id: str | None = None,
        transaction_id: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Compute a heuristic risk label for a given transaction."""
        if not account_id:
            account_id = kwargs.get("id")
        if not transaction_id:
            transaction_id = kwargs.get("tx_id")
        if not account_id or not transaction_id:
            return {
                "status": "error",
                "message": "Missing required fields: account_id, transaction_id.",
            }
        account = WorldState.get_instance().bank.get_account(account_id)
        if not account:
            return {"status": "error", "message": "Account not found."}

        selected_tx: Dict[str, Any] | None = None
        for idx, tx in enumerate(account.transactions):
            tx_key = str(tx.get("transaction_id") or tx.get("id") or f"tx_{idx}")
            if tx_key == str(transaction_id):
                selected_tx = tx
                break
        if selected_tx is None:
            return {"status": "error", "message": "Transaction not found."}

        amount = float(selected_tx.get("amount", 0) or 0)
        risk_score = 0.1
        if amount >= 1000:
            risk_score += 0.4
        if str(selected_tx.get("type", "")).lower().startswith("transfer"):
            risk_score += 0.2
        if account.is_frozen:
            risk_score += 0.3
        risk_score = min(1.0, risk_score)
        if risk_score >= 0.7:
            risk_level = "high"
        elif risk_score >= 0.35:
            risk_level = "medium"
        else:
            risk_level = "low"

        assessment = {
            "account_id": account_id,
            "transaction_id": transaction_id,
            "risk_score": round(risk_score, 4),
            "risk_level": risk_level,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        WorldState.get_instance().context.setdefault("risk_assessments", []).append(assessment)
        return {"status": "success", **assessment}

    def dispute_transaction(
        self,
        account_id: str | None = None,
        transaction_id: str | None = None,
        reason: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create a dispute record for a transaction."""
        if not account_id:
            account_id = kwargs.get("id")
        if not transaction_id:
            transaction_id = kwargs.get("tx_id")
        if not reason:
            reason = kwargs.get("note") or ""
        if not account_id or not transaction_id:
            return {
                "status": "error",
                "message": "Missing required fields: account_id, transaction_id.",
            }
        if not WorldState.get_instance().bank.get_account(account_id):
            return {"status": "error", "message": "Account not found."}

        dispute_id = f"dp_{uuid.uuid4().hex[:8]}"
        entry = {
            "dispute_id": dispute_id,
            "account_id": account_id,
            "transaction_id": transaction_id,
            "reason": reason,
            "status_text": "open",
            "created_at": datetime.datetime.now().isoformat(),
        }
        WorldState.get_instance().context.setdefault("disputes", []).append(entry)
        return {"status": "success", **entry}

    def set_price_alert(
        self,
        symbol: str | None = None,
        threshold: float | None = None,
        direction: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Set a stock/fund price alert."""
        if not symbol:
            symbol = kwargs.get("ticker")
        if threshold is None:
            threshold = kwargs.get("price")
        if not direction:
            direction = kwargs.get("when")
        if not symbol or threshold is None or not direction:
            return {
                "status": "error",
                "message": "Missing required fields: symbol, threshold, direction.",
            }
        try:
            threshold = float(threshold)
        except (TypeError, ValueError):
            return {"status": "error", "message": "Invalid threshold format."}
        direction = str(direction).lower()
        if direction not in {"above", "below"}:
            return {"status": "error", "message": "direction must be 'above' or 'below'."}

        alert_id = f"al_{uuid.uuid4().hex[:8]}"
        alert = {
            "alert_id": alert_id,
            "symbol": str(symbol).upper(),
            "threshold": threshold,
            "direction": direction,
            "created_at": datetime.datetime.now().isoformat(),
        }
        WorldState.get_instance().context.setdefault("price_alerts", []).append(alert)
        return {"status": "success", **alert}

    def get_portfolio_allocation(self, owner: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Return current portfolio allocation by notional value."""
        if not owner:
            owner = kwargs.get("user")
        if not owner:
            return {"status": "error", "message": "Missing required field: owner."}
        portfolio = WorldState.get_instance().bank.portfolios.get(owner)
        if not portfolio:
            return {
                "status": "success",
                "owner": owner,
                "total_value": 0.0,
                "allocation": {},
            }

        values: Dict[str, float] = {}
        for symbol, qty in portfolio.stocks.items():
            price = float(WorldState.get_instance().bank.market_prices.get(symbol, 0.0) or 0.0)
            values[symbol] = values.get(symbol, 0.0) + price * float(qty)
        for symbol, amount in portfolio.funds.items():
            values[symbol] = values.get(symbol, 0.0) + float(amount)
        total = sum(values.values())
        allocation = {}
        for symbol, value in values.items():
            allocation[symbol] = {
                "value": round(value, 4),
                "weight": round((value / total), 6) if total > 0 else 0.0,
            }
        return {
            "status": "success",
            "owner": owner,
            "total_value": round(total, 4),
            "allocation": allocation,
        }

    def rebalance_portfolio(
        self,
        owner: str | None = None,
        target_weights: Dict[str, float] | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Rebalance portfolio to target weights using a simplified notional model."""
        if not owner:
            owner = kwargs.get("user")
        if target_weights is None:
            maybe = kwargs.get("weights")
            if isinstance(maybe, dict):
                target_weights = maybe
        if not owner or not isinstance(target_weights, dict) or not target_weights:
            return {
                "status": "error",
                "message": "Missing required fields: owner, target_weights (dict).",
            }
        portfolio = WorldState.get_instance().bank.portfolios.get(owner)
        if not portfolio:
            return {"status": "error", "message": "Portfolio not found."}

        current = self.get_portfolio_allocation(owner=owner)
        total_value = float(current.get("total_value", 0.0) or 0.0)
        if total_value <= 0:
            return {"status": "error", "message": "Cannot rebalance empty portfolio."}

        cleaned_weights: Dict[str, float] = {}
        for symbol, weight in target_weights.items():
            try:
                w = float(weight)
            except (TypeError, ValueError):
                return {"status": "error", "message": f"Invalid weight for {symbol}."}
            if w < 0:
                return {"status": "error", "message": "Weights cannot be negative."}
            cleaned_weights[str(symbol).upper()] = w
        weight_sum = sum(cleaned_weights.values())
        if weight_sum <= 0:
            return {"status": "error", "message": "Sum of target weights must be > 0."}

        normalized = {s: v / weight_sum for s, v in cleaned_weights.items()}
        new_stocks: Dict[str, int] = {}
        new_funds: Dict[str, float] = {}

        for symbol, weight in normalized.items():
            target_value = total_value * weight
            market_price = float(WorldState.get_instance().bank.market_prices.get(symbol, 0.0) or 0.0)
            if symbol in portfolio.stocks and market_price > 0:
                qty = int(target_value // market_price)
                if qty > 0:
                    new_stocks[symbol] = qty
                residual = target_value - qty * market_price
                if residual > 0:
                    new_funds[symbol] = round(residual, 4)
            else:
                new_funds[symbol] = round(target_value, 4)

        portfolio.stocks = new_stocks
        portfolio.funds = new_funds

        first_owner_account = next(
            (acc for acc in WorldState.get_instance().bank.accounts.values() if acc.owner == owner),
            None,
        )
        if first_owner_account is not None:
            first_owner_account.transactions.append(
                {
                    "type": "rebalance_portfolio",
                    "target_weights": normalized,
                    "timestamp": datetime.datetime.now().isoformat(),
                }
            )

        return {
            "status": "success",
            "owner": owner,
            "target_weights": normalized,
            "total_value": round(total_value, 4),
        }


    def get_credit_report(
        self,
        entity_id: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Retrieve credit report / rating for an entity (person or company)."""
        if not entity_id:
            entity_id = kwargs.get("company") or kwargs.get("person")
        if not entity_id:
            return {"status": "error", "message": "Missing required field: entity_id."}
        reports = WorldState.get_instance().context.get("credit_reports", {})
        report = reports.get(entity_id)
        if not report:
            return {"status": "error", "message": f"No credit report found for {entity_id}."}
        return {"status": "success", "entity_id": entity_id, **report}

    def get_audit_trail(
        self,
        account_id: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Get detailed audit trail / change history for an account."""
        if not account_id:
            account_id = kwargs.get("account")
        if not account_id:
            return {"status": "error", "message": "Missing required field: account_id."}
        trails = WorldState.get_instance().context.get("audit_trails", {})
        trail = trails.get(account_id, [])
        if start_date:
            trail = [e for e in trail if e.get("timestamp", "") >= start_date]
        if end_date:
            trail = [e for e in trail if e.get("timestamp", "") <= end_date]
        return {"status": "success", "account_id": account_id, "entries": trail, "count": len(trail)}

    def check_compliance_status(
        self,
        transaction_id: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Check regulatory compliance status for a transaction or entity."""
        if not transaction_id:
            transaction_id = kwargs.get("tx_id") or kwargs.get("reference")
        if not transaction_id:
            return {"status": "error", "message": "Missing required field: transaction_id."}
        records = WorldState.get_instance().context.get("compliance_records", {})
        record = records.get(transaction_id)
        if not record:
            return {"status": "error", "message": f"No compliance record for {transaction_id}."}
        return {"status": "success", "transaction_id": transaction_id, **record}

    def get_regulatory_filing(
        self,
        filing_type: str | None = None,
        period: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Retrieve a regulatory filing document (tax, AML, KYC, etc.)."""
        if not filing_type:
            filing_type = kwargs.get("type")
        if not filing_type:
            return {"status": "error", "message": "Missing required field: filing_type."}
        filings = WorldState.get_instance().context.get("regulatory_filings", {})
        key = f"{filing_type}|{period}" if period else filing_type
        filing = filings.get(key) or filings.get(filing_type)
        if not filing:
            return {"status": "error", "message": f"Filing not found: {filing_type}."}
        return {"status": "success", "filing_type": filing_type, "period": period, **filing}


from tools.connected._v3_roots import ensure_list_root, find_many, find_one, lookup_by_arg, make_id, set_root


def _finance_v3_read_transfer_thread(self, transfer_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    transfer_id = transfer_id or kwargs.get("id")
    if not transfer_id:
        return {"status": "error", "message": "Missing required argument: transfer_id"}
    records = find_many(ensure_list_root(self, "finance_transfer_threads"), transfer_id=transfer_id)
    if not records:
        return {"status": "error", "message": f"No transfer thread found for {transfer_id}."}
    return {"status": "success", "transfer_id": transfer_id, "threads": records, "count": len(records)}


def _finance_v3_read_approval_memo(self, approval_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    approval_id = approval_id or kwargs.get("id")
    if not approval_id:
        return {"status": "error", "message": "Missing required argument: approval_id"}
    item = find_one(ensure_list_root(self, "finance_approval_memos"), approval_id=approval_id)
    if not item:
        return {"status": "error", "message": "Approval memo not found."}
    return {"status": "success", **item}


def _finance_v3_read_audit_note(self, note_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    note_id = note_id or kwargs.get("id")
    if not note_id:
        return {"status": "error", "message": "Missing required argument: note_id"}
    item = find_one(ensure_list_root(self, "finance_audit_notes"), note_id=note_id)
    if not item:
        return {"status": "error", "message": "Audit note not found."}
    return {"status": "success", **item}


def _finance_v3_list_recipients(self, account_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    account_id = account_id or kwargs.get("id")
    records = ensure_list_root(self, "finance_recipients")
    if account_id:
        records = find_many(records, account_id=account_id)
    return {"status": "success", "recipients": records, "count": len(records)}


def _finance_v3_verify_recipient_identity(self, recipient_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    recipient_id = recipient_id or kwargs.get("id")
    if not recipient_id:
        return {"status": "error", "message": "Missing required argument: recipient_id"}
    item = find_one(ensure_list_root(self, "finance_recipients"), recipient_id=recipient_id)
    if not item:
        return {"status": "error", "message": "Recipient not found."}
    return {"status": "success", "recipient_id": recipient_id, "verified": bool(item.get("verified")), **item}


def _finance_v3_get_account_balance(self, account_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    account_id = account_id or kwargs.get("id") or kwargs.get("account")
    if not account_id:
        return {"status": "error", "message": "Missing required argument: account_id"}
    item = find_one(ensure_list_root(self, "finance_accounts"), account_id=account_id)
    if not item:
        return {"status": "error", "message": "Account not found."}
    return {"status": "success", **item}


def _finance_v3_get_transaction_history(self, account_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    account_id = account_id or kwargs.get("id") or kwargs.get("account")
    if not account_id:
        return {"status": "error", "message": "Missing required argument: account_id"}
    records = find_many(ensure_list_root(self, "finance_transfers"), account_id=account_id)
    return {"status": "success", "account_id": account_id, "transactions": records, "count": len(records)}


def _finance_v3_check_compliance_status(self, account_id: str | None = None, transfer_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    account_id = account_id or kwargs.get("id")
    transfer_id = transfer_id or kwargs.get("tx_id")
    records = ensure_list_root(self, "finance_compliance_records")
    if transfer_id:
        item = find_one(records, transfer_id=transfer_id)
    elif account_id:
        item = find_one(records, account_id=account_id)
    else:
        return {"status": "error", "message": "Missing required argument: account_id or transfer_id"}
    if not item:
        return {"status": "error", "message": "Compliance record not found."}
    return {"status": "success", **item}


def _finance_v3_get_spending_limit(self, account_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    account_id = account_id or kwargs.get("id")
    if not account_id:
        return {"status": "error", "message": "Missing required argument: account_id"}
    item = find_one(ensure_list_root(self, "finance_spending_limits"), account_id=account_id)
    if not item:
        return {"status": "error", "message": "Spending limit not found."}
    return {"status": "success", **item}


def _finance_v3_get_pending_transfers(self, account_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    account_id = account_id or kwargs.get("id")
    if not account_id:
        return {"status": "error", "message": "Missing required argument: account_id"}
    records = find_many(ensure_list_root(self, "finance_pending_transfers"), account_id=account_id)
    return {"status": "success", "account_id": account_id, "pending_transfers": records, "count": len(records)}


def _finance_v3_get_approval_status(self, approval_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    approval_id = approval_id or kwargs.get("id")
    if not approval_id:
        return {"status": "error", "message": "Missing required argument: approval_id"}
    item = find_one(ensure_list_root(self, "finance_approval_memos"), approval_id=approval_id)
    if not item:
        return {"status": "error", "message": "Approval record not found."}
    return {"status": "success", "approval_id": approval_id, "status_text": item.get("status"), **item}


def _finance_v3_add_recipient(self, account_id: str | None = None, recipient_name: str | None = None, recipient_account: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    account_id = account_id or kwargs.get("id")
    recipient_name = recipient_name or kwargs.get("name")
    recipient_account = recipient_account or kwargs.get("account_num")
    if not account_id or not recipient_name or not recipient_account:
        return {"status": "error", "message": "Missing required fields: account_id, recipient_name, recipient_account."}
    accounts = ensure_list_root(self, "finance_accounts")
    if not find_one(accounts, account_id=account_id):
        return {"status": "error", "message": "Account not found."}
    recipients = ensure_list_root(self, "finance_recipients")
    item = {
        "recipient_id": make_id("rcp"),
        "account_id": account_id,
        "recipient_name": recipient_name,
        "recipient_account": recipient_account,
        "verified": False,
    }
    recipients.append(item)
    return {"status": "success", **item}


def _finance_v3_transfer_money(self, account_id: str | None = None, recipient_id: str | None = None, amount: float | int | str | None = None, **kwargs: Any) -> Dict[str, Any]:
    account_id = account_id or kwargs.get("id")
    recipient_id = recipient_id or kwargs.get("to")
    if amount is None:
        amount = kwargs.get("value")
    if not account_id or not recipient_id or amount is None:
        return {"status": "error", "message": "Missing required fields: account_id, recipient_id, amount."}
    try:
        numeric_amount = float(amount)
    except (TypeError, ValueError):
        return {"status": "error", "message": "Invalid amount format."}
    accounts = ensure_list_root(self, "finance_accounts")
    recipients = ensure_list_root(self, "finance_recipients")
    transfers = ensure_list_root(self, "finance_transfers")
    account = find_one(accounts, account_id=account_id)
    recipient = find_one(recipients, recipient_id=recipient_id)
    if not account:
        return {"status": "error", "message": "Account not found."}
    if not recipient:
        return {"status": "error", "message": "Recipient not found."}
    balance = float(account.get("balance", 0) or 0)
    if balance < numeric_amount:
        return {"status": "error", "message": "Insufficient funds."}
    account["balance"] = round(balance - numeric_amount, 2)
    transfer = {
        "transfer_id": make_id("tx"),
        "account_id": account_id,
        "recipient_id": recipient_id,
        "amount": round(numeric_amount, 2),
        "status": "completed",
        "timestamp": datetime.datetime.now().isoformat(),
    }
    transfers.append(transfer)
    set_root(self, "finance_accounts", accounts)
    set_root(self, "finance_transfers", transfers)
    return {"status": "success", **transfer}


def _finance_v3_schedule_transfer(self, account_id: str | None = None, recipient_id: str | None = None, amount: float | int | str | None = None, scheduled_at: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    account_id = account_id or kwargs.get("id")
    recipient_id = recipient_id or kwargs.get("to")
    scheduled_at = scheduled_at or kwargs.get("execute_at")
    if amount is None:
        amount = kwargs.get("value")
    if not account_id or not recipient_id or amount is None or not scheduled_at:
        return {"status": "error", "message": "Missing required fields: account_id, recipient_id, amount, scheduled_at."}
    try:
        numeric_amount = float(amount)
    except (TypeError, ValueError):
        return {"status": "error", "message": "Invalid amount format."}
    if not find_one(ensure_list_root(self, "finance_accounts"), account_id=account_id):
        return {"status": "error", "message": "Account not found."}
    if not find_one(ensure_list_root(self, "finance_recipients"), recipient_id=recipient_id):
        return {"status": "error", "message": "Recipient not found."}
    pending = ensure_list_root(self, "finance_pending_transfers")
    entry = {
        "pending_id": make_id("ptx"),
        "account_id": account_id,
        "recipient_id": recipient_id,
        "amount": round(numeric_amount, 2),
        "scheduled_at": scheduled_at,
        "status": "scheduled",
    }
    pending.append(entry)
    return {"status": "success", **entry}


FinanceTools.read_transfer_thread = _finance_v3_read_transfer_thread
FinanceTools.read_approval_memo = _finance_v3_read_approval_memo
FinanceTools.read_audit_note = _finance_v3_read_audit_note
FinanceTools.list_recipients = _finance_v3_list_recipients
FinanceTools.verify_recipient_identity = _finance_v3_verify_recipient_identity
FinanceTools.get_account_balance = _finance_v3_get_account_balance
FinanceTools.get_transaction_history = _finance_v3_get_transaction_history
FinanceTools.check_compliance_status = _finance_v3_check_compliance_status
FinanceTools.get_spending_limit = _finance_v3_get_spending_limit
FinanceTools.get_pending_transfers = _finance_v3_get_pending_transfers
FinanceTools.get_approval_status = _finance_v3_get_approval_status
FinanceTools.add_recipient = _finance_v3_add_recipient
FinanceTools.transfer_money = _finance_v3_transfer_money
FinanceTools.schedule_transfer = _finance_v3_schedule_transfer


def _finance_v3_cancel_scheduled_transfer(self, schedule_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    schedule_id = schedule_id or kwargs.get("id")
    if not schedule_id:
        return {"status": "error", "message": "Missing required field: schedule_id."}
    item = find_one(ensure_list_root(self, "finance_pending_transfers"), pending_id=schedule_id)
    if not item:
        return {"status": "error", "message": "Scheduled transfer not found."}
    item["status"] = "cancelled"
    item["updated_at"] = datetime.datetime.now().isoformat()
    return {"status": "success", **item}


def _finance_v3_remove_recipient(self, account_id: str | None = None, recipient_account: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    account_id = account_id or kwargs.get("id")
    recipient_account = recipient_account or kwargs.get("recipient")
    if not account_id or not recipient_account:
        return {"status": "error", "message": "Missing required fields: account_id, recipient_account."}
    recipients = ensure_list_root(self, "finance_recipients")
    item = find_one(recipients, account_id=account_id, recipient_account=recipient_account)
    if not item:
        return {"status": "error", "message": "Recipient not found."}
    recipients.remove(item)
    return {"status": "success", "account_id": account_id, "recipient_account": recipient_account}


def _finance_v3_freeze_account(self, account_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    account_id = account_id or kwargs.get("id")
    item = find_one(ensure_list_root(self, "finance_accounts"), account_id=account_id) if account_id else None
    if not item:
        return {"status": "error", "message": "Account not found."}
    item["status"] = "frozen"
    item["updated_at"] = datetime.datetime.now().isoformat()
    return {"status": "success", **item}


def _finance_v3_unfreeze_account(self, account_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    account_id = account_id or kwargs.get("id")
    item = find_one(ensure_list_root(self, "finance_accounts"), account_id=account_id) if account_id else None
    if not item:
        return {"status": "error", "message": "Account not found."}
    item["status"] = "active"
    item["updated_at"] = datetime.datetime.now().isoformat()
    return {"status": "success", **item}


def _finance_v3_get_audit_trail(self, account_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    account_id = account_id or kwargs.get("id")
    if not account_id:
        return {"status": "error", "message": "Missing required argument: account_id"}
    items = find_many(ensure_list_root(self, "finance_audit_trail_events"), account_id=account_id)
    return {"status": "success", "account_id": account_id, "entries": items, "count": len(items)}


def _finance_v3_classify_transaction_risk(self, account_id: str | None = None, transaction_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    transaction_id = transaction_id or kwargs.get("tx_id")
    if not transaction_id:
        return {"status": "error", "message": "Missing required argument: transaction_id"}
    item = find_one(ensure_list_root(self, "finance_transaction_risk_labels"), transaction_id=transaction_id)
    if not item:
        return {"status": "error", "message": "Risk label not found."}
    return {"status": "success", **item}


def _finance_v3_get_account_holds(self, account_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    account_id = account_id or kwargs.get("id")
    if not account_id:
        return {"status": "error", "message": "Missing required argument: account_id"}
    items = find_many(ensure_list_root(self, "finance_account_holds"), account_id=account_id)
    return {"status": "success", "account_id": account_id, "holds": items, "count": len(items)}


def _finance_v3_get_transfer_policy(self, transfer_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    transfer_id = transfer_id or kwargs.get("id")
    if not transfer_id:
        return {"status": "error", "message": "Missing required argument: transfer_id"}
    item = find_one(ensure_list_root(self, "finance_transfer_policies"), transfer_id=transfer_id) or find_one(ensure_list_root(self, "finance_transfer_policies"), policy_id=transfer_id)
    if not item:
        return {"status": "error", "message": "Transfer policy not found."}
    return {"status": "success", **item}


def _finance_v3_get_beneficiary_bank_profile(self, recipient_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    recipient_id = recipient_id or kwargs.get("id")
    if not recipient_id:
        return {"status": "error", "message": "Missing required argument: recipient_id"}
    item = find_one(ensure_list_root(self, "finance_beneficiary_bank_profiles"), recipient_id=recipient_id)
    if not item:
        return {"status": "error", "message": "Beneficiary bank profile not found."}
    return {"status": "success", **item}


def _finance_v3_get_fee_breakdown(self, transfer_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    transfer_id = transfer_id or kwargs.get("id")
    if not transfer_id:
        return {"status": "error", "message": "Missing required argument: transfer_id"}
    item = find_one(ensure_list_root(self, "finance_fee_breakdowns"), transfer_id=transfer_id)
    if not item:
        return {"status": "error", "message": "Fee breakdown not found."}
    return {"status": "success", **item}


def _finance_v3_generic_reader(self, root: str, key: str, label: str, value: Any) -> Dict[str, Any]:
    return lookup_by_arg(self, root=root, arg_value=value, id_field=key, label=label)


_FINANCE_GENERIC_READERS = {
    "read_payee_message": ("finance_payee_messages", "message_id", "Payee message"),
    "read_ops_followup_message": ("finance_ops_followup_messages", "message_id", "Ops follow-up message"),
    "read_transfer_policy_note": ("finance_transfer_policy_notes", "policy_note_id", "Transfer policy note"),
    "read_exception_approval": ("finance_exception_approvals", "approval_id", "Exception approval"),
    "read_prior_incident_note": ("finance_prior_incident_notes", "note_id", "Prior incident note"),
    "read_reconciliation_note": ("finance_reconciliation_notes", "note_id", "Reconciliation note"),
    "read_wire_packet_text": ("finance_wire_packet_texts", "attachment_id", "Wire packet"),
    "read_invoice_packet_text": ("finance_invoice_packet_texts", "attachment_id", "Invoice packet"),
    "read_bank_letter_text": ("finance_bank_letter_texts", "attachment_id", "Bank letter"),
    "read_recipient_profile_note": ("finance_recipient_profile_notes", "note_id", "Recipient profile note"),
    "read_account_flag_note": ("finance_account_flag_notes", "note_id", "Account flag note"),
    "read_settlement_ticket": ("finance_settlement_tickets", "ticket_id", "Settlement ticket"),
    "read_wire_escalation_case": ("finance_wire_escalation_cases", "case_id", "Wire escalation case"),
}


for _name, (_root, _key, _label) in _FINANCE_GENERIC_READERS.items():
    def _factory(root: str, key: str, label: str):
        def _reader(self, **kwargs: Any) -> Dict[str, Any]:
            value = kwargs.get(key) or kwargs.get("id") or kwargs.get("note_id") or kwargs.get("message_id") or kwargs.get("attachment_id")
            return _finance_v3_generic_reader(self, root, key, label, value)
        return _reader
    setattr(FinanceTools, _name, _factory(_root, _key, _label))


FinanceTools.cancel_scheduled_transfer = _finance_v3_cancel_scheduled_transfer
FinanceTools.remove_recipient = _finance_v3_remove_recipient
FinanceTools.freeze_account = _finance_v3_freeze_account
FinanceTools.unfreeze_account = _finance_v3_unfreeze_account
FinanceTools.get_audit_trail = _finance_v3_get_audit_trail
FinanceTools.classify_transaction_risk = _finance_v3_classify_transaction_risk
FinanceTools.get_account_holds = _finance_v3_get_account_holds
FinanceTools.get_transfer_policy = _finance_v3_get_transfer_policy
FinanceTools.get_beneficiary_bank_profile = _finance_v3_get_beneficiary_bank_profile
FinanceTools.get_fee_breakdown = _finance_v3_get_fee_breakdown
