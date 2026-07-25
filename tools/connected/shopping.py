from typing import Any, Dict, List
import uuid
import datetime
from core.world.environment import WorldState, Product, Order

class ShoppingTools:
    """
    Shopping tools connected to the World Model.
    Handles E-commerce operations.
    """

    def __init__(self):
        self.current_user = "user@example.com"

    def search_product(self, query: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Search for products by name or category."""
        if not query:
            query = kwargs.get("keyword") or kwargs.get("name")
        if not query:
            return {"status": "error", "message": "Missing required argument: query"}

        results = []
        for p in WorldState.get_instance().shopping.products.values():
            if query.lower() in p.name.lower() or query.lower() in p.category.lower():
                results.append({
                    "id": p.id,
                    "name": p.name,
                    "price": p.price,
                    "in_stock": p.stock > 0
                })
        return {"status": "success", "products": results, "count": len(results)}

    def get_product_details(self, product_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Get detailed info for a product."""
        if not product_id:
            product_id = kwargs.get("id")
        if not product_id:
            return {"status": "error", "message": "Missing required argument: product_id"}

        p = WorldState.get_instance().shopping.products.get(product_id)
        if not p:
            return {"status": "error", "message": "Product not found."}
        return {
            "status": "success",
            "id": p.id,
            "name": p.name,
            "price": p.price,
            "description": p.description,
            "stock": p.stock,
            "category": p.category
        }

    def add_to_cart(
        self,
        product_id: str | None = None,
        quantity: int = 1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Add item to user's cart."""
        if not product_id:
            product_id = kwargs.get("id")
        if quantity == 1 and kwargs.get("qty") is not None:
            quantity = kwargs.get("qty")
        if not product_id:
            return {"status": "error", "message": "Missing required argument: product_id"}
        try:
            quantity = int(quantity)
        except (TypeError, ValueError):
            return {"status": "error", "message": "Invalid quantity format."}
        if quantity <= 0:
            return {"status": "error", "message": "Quantity must be positive."}

        p = WorldState.get_instance().shopping.products.get(product_id)
        if not p:
            return {"status": "error", "message": "Product not found."}
        if p.stock < quantity:
            return {"status": "error", "message": f"Insufficient stock. Available: {p.stock}"}

        cart = WorldState.get_instance().shopping.carts.get(self.current_user, [])

        existing_item = next((item for item in cart if item["product_id"] == product_id), None)
        if existing_item:
            existing_item["quantity"] += quantity
        else:
            cart.append({
                "product_id": product_id,
                "name": p.name,
                "price": p.price,
                "quantity": quantity
            })

        WorldState.get_instance().shopping.carts[self.current_user] = cart
        return {"status": "success", "message": f"Added {quantity} x {p.name} to cart."}

    def view_cart(self) -> Dict[str, Any]:
        """View current cart."""
        cart = WorldState.get_instance().shopping.carts.get(self.current_user, [])
        total = sum(item["price"] * item["quantity"] for item in cart)
        return {
            "status": "success",
            "items": cart,
            "total_price": total
        }

    def place_order(
        self,
        product_id: str | None = None,
        quantity: int = 1,
        address: str | None = None,
        shipping_address: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Checkout: optionally add product_id/quantity first, then place order."""
        if not address:
            address = shipping_address or kwargs.get("ship_to") or kwargs.get("address")
        if not address:
            address = "default"

        if product_id or kwargs.get("id"):
            pid = product_id or kwargs.get("id")
            qty = int(kwargs.get("qty", quantity) or quantity)
            result = self.add_to_cart(product_id=pid, quantity=qty)
            if result["status"] == "error":
                return result

        cart = WorldState.get_instance().shopping.carts.get(self.current_user, [])
        if not cart:
            return {"status": "error", "message": "Cart is empty. Add items first with add_to_cart."}

        for item in cart:
            p = WorldState.get_instance().shopping.products.get(item["product_id"])
            if not p or p.stock < item["quantity"]:
                return {"status": "error", "message": f"Product {item['name']} is out of stock."}

        for item in cart:
             p = WorldState.get_instance().shopping.products.get(item["product_id"])
             p.stock -= item["quantity"]

        total = sum(item["price"] * item["quantity"] for item in cart)
        order_id = f"ord_{uuid.uuid4().hex[:8]}"
        order = Order(
            id=order_id,
            user=self.current_user,
            items=cart,
            total_price=total,
            status="Processing"
        )
        WorldState.get_instance().shopping.orders.append(order)

        WorldState.get_instance().shopping.carts[self.current_user] = []

        return {
            "status": "success",
            "message": "Order placed successfully.",
            "order_id": order_id,
            "total_price": total
        }

    def track_order(self, order_id: str) -> Dict[str, Any]:
        order = next((o for o in WorldState.get_instance().shopping.orders if o.id == order_id), None)
        if not order:
            return {"status": "error", "message": "Order not found."}
        return {
            "status": "success",
            "order_id": order.id,
            "order_status": order.status,
        }

    def get_recommendations(self, limit: int = 5, **kwargs: Any) -> Dict[str, Any]:
        """Return simple recommendations based on in-stock products."""
        if kwargs.get("top_k") is not None:
            limit = kwargs.get("top_k")
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 5
        limit = max(1, min(limit, 20))

        products = [
            {
                "id": p.id,
                "name": p.name,
                "price": p.price,
                "category": p.category,
            }
            for p in WorldState.get_instance().shopping.products.values()
            if p.stock > 0
        ]
        products.sort(key=lambda x: x["price"])
        return {"status": "success", "products": products[:limit], "count": min(limit, len(products))}

    def remove_from_cart(self, product_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Remove an item from cart."""
        if not product_id:
            product_id = kwargs.get("id")
        if not product_id:
            return {"status": "error", "message": "Missing required argument: product_id"}

        cart = WorldState.get_instance().shopping.carts.get(self.current_user, [])
        old_len = len(cart)
        cart = [item for item in cart if item.get("product_id") != product_id]
        WorldState.get_instance().shopping.carts[self.current_user] = cart
        if len(cart) == old_len:
            return {"status": "error", "message": "Product not found in cart."}
        return {"status": "success", "message": f"Removed {product_id} from cart."}

    def return_item(self, order_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Initiate an order return."""
        if not order_id:
            order_id = kwargs.get("id")
        if not order_id:
            return {"status": "error", "message": "Missing required argument: order_id"}

        order = next((o for o in WorldState.get_instance().shopping.orders if o.id == order_id), None)
        if not order:
            return {"status": "error", "message": "Order not found."}
        if order.user != self.current_user:
            return {"status": "error", "message": "Access denied."}

        order.status = "Returned"
        return {
            "status": "success",
            "order_id": order.id,
            "order_status": order.status,
        }

    def add_to_wishlist(self, product_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Add a product to wishlist."""
        if not product_id:
            product_id = kwargs.get("id")
        if not product_id:
            return {"status": "error", "message": "Missing required field: product_id."}
        if product_id not in WorldState.get_instance().shopping.products:
            return {"status": "error", "message": "Product not found."}
        wishlists = WorldState.get_instance().context.setdefault("wishlists", {})
        items = wishlists.setdefault(self.current_user, [])
        if product_id not in items:
            items.append(product_id)
        return {"status": "success", "wishlist": items}

    def remove_from_wishlist(self, product_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Remove a product from wishlist."""
        if not product_id:
            product_id = kwargs.get("id")
        if not product_id:
            return {"status": "error", "message": "Missing required field: product_id."}
        wishlists = WorldState.get_instance().context.setdefault("wishlists", {})
        items = wishlists.setdefault(self.current_user, [])
        if product_id not in items:
            return {"status": "error", "message": "Product not in wishlist."}
        wishlists[self.current_user] = [x for x in items if x != product_id]
        return {"status": "success", "wishlist": wishlists[self.current_user]}

    def view_wishlist(self, **kwargs: Any) -> Dict[str, Any]:
        """View current wishlist with product details."""
        wishlists = WorldState.get_instance().context.setdefault("wishlists", {})
        ids = wishlists.setdefault(self.current_user, [])
        products = []
        for pid in ids:
            p = WorldState.get_instance().shopping.products.get(pid)
            if p:
                products.append({"id": p.id, "name": p.name, "price": p.price, "category": p.category})
        return {"status": "success", "items": products, "count": len(products)}

    def apply_coupon(self, coupon_code: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Apply coupon to current user's cart context."""
        if not coupon_code:
            coupon_code = kwargs.get("code")
        if not coupon_code:
            return {"status": "error", "message": "Missing required field: coupon_code."}
        coupons = WorldState.get_instance().context.setdefault("cart_coupons", {})
        user_coupons = coupons.setdefault(self.current_user, [])
        if coupon_code not in user_coupons:
            user_coupons.append(coupon_code)
        return {"status": "success", "applied_coupons": list(user_coupons)}

    def remove_coupon(self, coupon_code: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Remove coupon from current user's cart context."""
        if not coupon_code:
            coupon_code = kwargs.get("code")
        if not coupon_code:
            return {"status": "error", "message": "Missing required field: coupon_code."}
        coupons = WorldState.get_instance().context.setdefault("cart_coupons", {})
        user_coupons = coupons.setdefault(self.current_user, [])
        if coupon_code not in user_coupons:
            return {"status": "error", "message": "Coupon not applied."}
        coupons[self.current_user] = [x for x in user_coupons if x != coupon_code]
        return {"status": "success", "applied_coupons": coupons[self.current_user]}

    def estimate_shipping(
        self,
        address: str | None = None,
        items: List[Dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Estimate shipping cost and ETA for selected items."""
        if not address:
            address = kwargs.get("shipping_address")
        if items is None and isinstance(kwargs.get("cart_items"), list):
            items = kwargs.get("cart_items")
        if not address:
            return {"status": "error", "message": "Missing required field: address."}
        if items is None:
            items = WorldState.get_instance().shopping.carts.get(self.current_user, [])

        total_units = 0
        for item in items:
            try:
                total_units += int(item.get("quantity", 1))
            except (TypeError, ValueError):
                total_units += 1
        base = 4.99
        shipping_fee = round(base + 1.5 * max(0, total_units - 1), 2)
        quote = {
            "address": address,
            "item_count": total_units,
            "shipping_fee": shipping_fee,
            "eta_days": max(1, min(10, 2 + total_units // 3)),
            "created_at": datetime.datetime.now().isoformat(),
        }
        WorldState.get_instance().context.setdefault("shipping_quotes", []).append(quote)
        return {"status": "success", **quote}

    def compare_products(self, product_ids: List[str] | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Compare products by key fields."""
        if product_ids is None and isinstance(kwargs.get("ids"), list):
            product_ids = kwargs.get("ids")
        if not isinstance(product_ids, list) or not product_ids:
            return {"status": "error", "message": "Missing required field: product_ids."}
        comparison = []
        for pid in product_ids:
            p = WorldState.get_instance().shopping.products.get(pid)
            if not p:
                continue
            comparison.append(
                {
                    "id": p.id,
                    "name": p.name,
                    "price": p.price,
                    "category": p.category,
                    "stock": p.stock,
                }
            )
        session_id = f"cmp_{uuid.uuid4().hex[:8]}"
        WorldState.get_instance().context.setdefault("compare_sessions", {})[session_id] = {
            "session_id": session_id,
            "product_ids": product_ids,
            "created_at": datetime.datetime.now().isoformat(),
        }
        return {"status": "success", "session_id": session_id, "comparison": comparison}

    def save_for_later(self, product_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Move cart item into save-for-later list."""
        if not product_id:
            product_id = kwargs.get("id")
        if not product_id:
            return {"status": "error", "message": "Missing required field: product_id."}
        cart = WorldState.get_instance().shopping.carts.get(self.current_user, [])
        idx = next((i for i, item in enumerate(cart) if item.get("product_id") == product_id), None)
        if idx is None:
            return {"status": "error", "message": "Product not found in cart."}
        item = cart.pop(idx)
        saved = WorldState.get_instance().context.setdefault("saved_for_later", {}).setdefault(self.current_user, [])
        saved.append(item)
        WorldState.get_instance().shopping.carts[self.current_user] = cart
        return {"status": "success", "saved_count": len(saved)}

    def move_to_cart_from_saved(self, product_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Move an item from saved-for-later back to cart."""
        if not product_id:
            product_id = kwargs.get("id")
        if not product_id:
            return {"status": "error", "message": "Missing required field: product_id."}
        saved = WorldState.get_instance().context.setdefault("saved_for_later", {}).setdefault(self.current_user, [])
        idx = next((i for i, item in enumerate(saved) if item.get("product_id") == product_id), None)
        if idx is None:
            return {"status": "error", "message": "Product not found in saved list."}
        item = saved.pop(idx)
        cart = WorldState.get_instance().shopping.carts.setdefault(self.current_user, [])
        existing = next((x for x in cart if x.get("product_id") == product_id), None)
        if existing:
            existing["quantity"] = int(existing.get("quantity", 0)) + int(item.get("quantity", 1))
        else:
            cart.append(item)
        return {"status": "success", "cart_count": len(cart)}

    def cancel_order(
        self,
        order_id: str | None = None,
        reason: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Cancel an existing order."""
        if not order_id:
            order_id = kwargs.get("id")
        if not reason:
            reason = kwargs.get("note") or ""
        if not order_id:
            return {"status": "error", "message": "Missing required field: order_id."}
        order = next((o for o in WorldState.get_instance().shopping.orders if o.id == order_id), None)
        if not order:
            return {"status": "error", "message": "Order not found."}
        if order.user != self.current_user:
            return {"status": "error", "message": "Access denied."}
        if order.status in {"Cancelled", "Returned"}:
            return {"status": "error", "message": f"Order already {order.status}."}
        order.status = "Cancelled"
        WorldState.get_instance().context.setdefault("order_cancellations", []).append(
            {
                "order_id": order_id,
                "reason": reason,
                "cancelled_at": datetime.datetime.now().isoformat(),
            }
        )
        return {
            "status": "success",
            "order_id": order.id,
            "order_status": order.status,
        }


    def check_supplier_rating(
        self,
        supplier_id: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Get reputation / quality rating for a supplier."""
        if not supplier_id:
            supplier_id = kwargs.get("supplier") or kwargs.get("vendor")
        if not supplier_id:
            return {"status": "error", "message": "Missing required field: supplier_id."}
        ratings = WorldState.get_instance().context.get("supplier_ratings", {})
        info = ratings.get(supplier_id)
        if not info:
            return {"status": "error", "message": f"No rating data for supplier {supplier_id}."}
        return {"status": "success", "supplier_id": supplier_id, **info}

    def get_price_history(
        self,
        product_id: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Get historical price data for a product."""
        if not product_id:
            product_id = kwargs.get("id") or kwargs.get("item")
        if not product_id:
            return {"status": "error", "message": "Missing required field: product_id."}
        history = WorldState.get_instance().context.get("price_history", {})
        data = history.get(product_id)
        if not data:
            return {"status": "error", "message": f"No price history for {product_id}."}
        return {"status": "success", "product_id": product_id, "history": data}

    def get_inventory_status(
        self,
        product_id: str | None = None,
        warehouse: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Check real-time inventory / stock levels for a product."""
        if not product_id:
            product_id = kwargs.get("id") or kwargs.get("item")
        if not product_id:
            return {"status": "error", "message": "Missing required field: product_id."}
        product = WorldState.get_instance().shopping.products.get(product_id)
        if not product:
            inv = WorldState.get_instance().context.get("inventory", {}).get(product_id)
            if inv:
                return {"status": "success", "product_id": product_id, **inv}
            return {"status": "error", "message": f"Product {product_id} not found."}
        return {
            "status": "success",
            "product_id": product_id,
            "name": product.name,
            "stock": product.stock,
            "warehouse": warehouse or "default",
        }


from tools.connected._v3_roots import ensure_list_root, find_many, find_one, lookup_by_arg, make_id


def _shopping_v3_read_order_message_thread(self, order_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    order_id = order_id or kwargs.get("id")
    if not order_id:
        return {"status": "error", "message": "Missing required argument: order_id"}
    items = find_many(ensure_list_root(self, "shopping_order_threads"), order_id=order_id)
    if not items:
        return {"status": "error", "message": f"No order thread found for {order_id}."}
    return {"status": "success", "order_id": order_id, "messages": items, "count": len(items)}


def _shopping_v3_read_return_policy_note(self, policy_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    policy_id = policy_id or kwargs.get("id")
    if not policy_id:
        return {"status": "error", "message": "Missing required argument: policy_id"}
    item = find_one(ensure_list_root(self, "shopping_return_policy_notes"), policy_id=policy_id)
    if not item:
        return {"status": "error", "message": "Return policy note not found."}
    return {"status": "success", **item}


def _shopping_v3_read_seller_remark(self, seller_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    seller_id = seller_id or kwargs.get("id")
    if not seller_id:
        return {"status": "error", "message": "Missing required argument: seller_id"}
    item = find_one(ensure_list_root(self, "shopping_seller_remarks"), seller_id=seller_id)
    if not item:
        return {"status": "error", "message": "Seller remark not found."}
    return {"status": "success", **item}


def _shopping_v3_track_order(self, order_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    order_id = order_id or kwargs.get("id")
    if not order_id:
        return {"status": "error", "message": "Missing required argument: order_id"}
    item = find_one(ensure_list_root(self, "shopping_orders"), order_id=order_id)
    if not item:
        return {"status": "error", "message": "Order not found."}
    return {"status": "success", "order_id": order_id, "status_text": item.get("status"), **item}


def _shopping_v3_get_order_details(self, order_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    order_id = order_id or kwargs.get("id")
    if not order_id:
        return {"status": "error", "message": "Missing required argument: order_id"}
    item = find_one(ensure_list_root(self, "shopping_orders"), order_id=order_id)
    if not item:
        return {"status": "error", "message": "Order not found."}
    return {"status": "success", **item}


def _shopping_v3_get_product_details(self, product_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    product_id = product_id or kwargs.get("id")
    if not product_id:
        return {"status": "error", "message": "Missing required argument: product_id"}
    item = find_one(ensure_list_root(self, "shopping_products"), product_id=product_id)
    if not item:
        return {"status": "error", "message": "Product not found."}
    return {"status": "success", **item}


def _shopping_v3_get_inventory_status(self, product_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    product_id = product_id or kwargs.get("id")
    if not product_id:
        return {"status": "error", "message": "Missing required argument: product_id"}
    item = find_one(ensure_list_root(self, "shopping_inventory_snapshots"), product_id=product_id)
    if not item:
        return {"status": "error", "message": "Inventory snapshot not found."}
    return {"status": "success", **item}


def _shopping_v3_estimate_shipping(self, order_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    order_id = order_id or kwargs.get("id")
    if not order_id:
        return {"status": "error", "message": "Missing required argument: order_id"}
    item = find_one(ensure_list_root(self, "shopping_shipping_quotes"), order_id=order_id)
    if not item:
        return {"status": "error", "message": "Shipping quote not found."}
    return {"status": "success", **item}


def _shopping_v3_check_supplier_rating(self, seller_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    seller_id = seller_id or kwargs.get("id")
    if not seller_id:
        return {"status": "error", "message": "Missing required argument: seller_id"}
    item = find_one(ensure_list_root(self, "shopping_supplier_ratings"), seller_id=seller_id)
    if not item:
        return {"status": "error", "message": "Supplier rating not found."}
    return {"status": "success", **item}


def _shopping_v3_compare_products(self, product_ids: list[str] | None = None, **kwargs: Any) -> Dict[str, Any]:
    product_ids = product_ids or kwargs.get("ids")
    if not isinstance(product_ids, list) or not product_ids:
        return {"status": "error", "message": "Missing required argument: product_ids"}
    products = ensure_list_root(self, "shopping_products")
    inventory = ensure_list_root(self, "shopping_inventory_snapshots")
    compared = []
    for product_id in product_ids:
        item = find_one(products, product_id=product_id)
        if not item:
            continue
        snap = find_one(inventory, product_id=product_id) or {}
        compared.append({**item, "inventory_status": snap.get("inventory_status"), "available_qty": snap.get("available_qty")})
    return {"status": "success", "products": compared, "count": len(compared)}


def _shopping_v3_place_order(self, cart_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    cart_id = cart_id or kwargs.get("id")
    if not cart_id:
        return {"status": "error", "message": "Missing required field: cart_id."}
    carts = ensure_list_root(self, "shopping_carts")
    orders = ensure_list_root(self, "shopping_orders")
    cart = find_one(carts, cart_id=cart_id)
    if not cart:
        return {"status": "error", "message": "Cart not found."}
    order = {
        "order_id": make_id("ord"),
        "user": cart.get("customer_id"),
        "status": "placed",
        "seller_id": cart.get("seller_id"),
        "items": list(cart.get("items", [])),
        "cart_id": cart_id,
        "created_at": datetime.datetime.now().isoformat(),
    }
    orders.append(order)
    cart["status"] = "checked_out"
    return {"status": "success", **order}


def _shopping_v3_cancel_order(self, order_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    order_id = order_id or kwargs.get("id")
    if not order_id:
        return {"status": "error", "message": "Missing required field: order_id."}
    order = find_one(ensure_list_root(self, "shopping_orders"), order_id=order_id)
    if not order:
        return {"status": "error", "message": "Order not found."}
    order["status"] = "cancelled"
    order["updated_at"] = datetime.datetime.now().isoformat()
    return {"status": "success", **order}


def _shopping_v3_return_item(self, order_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    order_id = order_id or kwargs.get("id")
    if not order_id:
        return {"status": "error", "message": "Missing required field: order_id."}
    order = find_one(ensure_list_root(self, "shopping_orders"), order_id=order_id)
    if not order:
        return {"status": "error", "message": "Order not found."}
    order["status"] = "return_requested"
    order["return_requested_at"] = datetime.datetime.now().isoformat()
    return {"status": "success", **order}


def _shopping_v3_apply_coupon(self, cart_id: str | None = None, coupon_code: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    cart_id = cart_id or kwargs.get("id")
    coupon_code = coupon_code or kwargs.get("code")
    if not cart_id or not coupon_code:
        return {"status": "error", "message": "Missing required fields: cart_id, coupon_code."}
    cart = find_one(ensure_list_root(self, "shopping_carts"), cart_id=cart_id)
    if not cart:
        return {"status": "error", "message": "Cart not found."}
    if not find_one(ensure_list_root(self, "shopping_coupon_rules"), coupon_code=coupon_code):
        return {"status": "error", "message": "Coupon not found."}
    codes = list(cart.get("coupon_codes", []))
    if coupon_code not in codes:
        codes.append(coupon_code)
    cart["coupon_codes"] = codes
    cart["updated_at"] = datetime.datetime.now().isoformat()
    return {"status": "success", "cart_id": cart_id, "coupon_codes": codes}


ShoppingTools.read_order_message_thread = _shopping_v3_read_order_message_thread
ShoppingTools.read_return_policy_note = _shopping_v3_read_return_policy_note
ShoppingTools.read_seller_remark = _shopping_v3_read_seller_remark
ShoppingTools.track_order = _shopping_v3_track_order
ShoppingTools.get_order_details = _shopping_v3_get_order_details
ShoppingTools.get_product_details = _shopping_v3_get_product_details
ShoppingTools.get_inventory_status = _shopping_v3_get_inventory_status
ShoppingTools.estimate_shipping = _shopping_v3_estimate_shipping
ShoppingTools.check_supplier_rating = _shopping_v3_check_supplier_rating
ShoppingTools.compare_products = _shopping_v3_compare_products
ShoppingTools.place_order = _shopping_v3_place_order
ShoppingTools.cancel_order = _shopping_v3_cancel_order
ShoppingTools.return_item = _shopping_v3_return_item
ShoppingTools.apply_coupon = _shopping_v3_apply_coupon


def _shopping_v3_remove_coupon(self, coupon_code: str | None = None, cart_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    cart_id = cart_id or kwargs.get("id")
    coupon_code = coupon_code or kwargs.get("code")
    if not cart_id or not coupon_code:
        return {"status": "error", "message": "Missing required fields: cart_id, coupon_code."}
    cart = find_one(ensure_list_root(self, "shopping_carts"), cart_id=cart_id)
    if not cart:
        return {"status": "error", "message": "Cart not found."}
    codes = [c for c in list(cart.get("coupon_codes", [])) if c != coupon_code]
    cart["coupon_codes"] = codes
    cart["updated_at"] = datetime.datetime.now().isoformat()
    return {"status": "success", "cart_id": cart_id, "coupon_codes": codes}


def _shopping_v3_remove_from_cart(self, product_id: str | None = None, cart_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    product_id = product_id or kwargs.get("id")
    cart_id = cart_id or kwargs.get("cart_id")
    carts = ensure_list_root(self, "shopping_carts")
    cart = find_one(carts, cart_id=cart_id) if cart_id else (carts[0] if carts else None)
    if not cart or not product_id:
        return {"status": "error", "message": "Missing required fields: product_id/cart."}
    items = [item for item in list(cart.get("items", [])) if item != product_id]
    cart["items"] = items
    cart["updated_at"] = datetime.datetime.now().isoformat()
    return {"status": "success", "cart_id": cart.get("cart_id"), "items": items}


def _shopping_v3_get_return_eligibility(self, order_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    order_id = order_id or kwargs.get("id")
    if not order_id:
        return {"status": "error", "message": "Missing required field: order_id."}
    item = find_one(ensure_list_root(self, "shopping_return_eligibility"), order_id=order_id)
    if not item:
        return {"status": "error", "message": "Return eligibility not found."}
    return {"status": "success", **item}


def _shopping_v3_get_coupon_eligibility(self, cart_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    cart_id = cart_id or kwargs.get("id")
    if not cart_id:
        return {"status": "error", "message": "Missing required field: cart_id."}
    item = find_one(ensure_list_root(self, "shopping_coupon_eligibility"), cart_id=cart_id)
    if not item:
        return {"status": "error", "message": "Coupon eligibility not found."}
    return {"status": "success", **item}


def _shopping_v3_get_cart_constraints(self, cart_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    cart_id = cart_id or kwargs.get("id")
    if not cart_id:
        return {"status": "error", "message": "Missing required field: cart_id."}
    item = find_one(ensure_list_root(self, "shopping_cart_constraints"), cart_id=cart_id)
    if not item:
        return {"status": "error", "message": "Cart constraints not found."}
    return {"status": "success", **item}


def _shopping_v3_get_delivery_window(self, order_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    order_id = order_id or kwargs.get("id")
    if not order_id:
        return {"status": "error", "message": "Missing required field: order_id."}
    item = find_one(ensure_list_root(self, "shopping_delivery_windows"), order_id=order_id)
    if not item:
        return {"status": "error", "message": "Delivery window not found."}
    return {"status": "success", **item}


def _shopping_v3_get_seller_policy(self, seller_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    seller_id = seller_id or kwargs.get("id")
    if not seller_id:
        return {"status": "error", "message": "Missing required field: seller_id."}
    item = find_one(ensure_list_root(self, "shopping_seller_policies"), seller_id=seller_id)
    if not item:
        return {"status": "error", "message": "Seller policy not found."}
    return {"status": "success", **item}


def _shopping_v3_get_restock_eta(self, product_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    product_id = product_id or kwargs.get("id")
    if not product_id:
        return {"status": "error", "message": "Missing required field: product_id."}
    item = find_one(ensure_list_root(self, "shopping_restock_eta"), product_id=product_id)
    if not item:
        return {"status": "error", "message": "Restock ETA not found."}
    return {"status": "success", **item}


def _shopping_v3_generic_reader(self, root: str, key: str, label: str, value: Any) -> Dict[str, Any]:
    return lookup_by_arg(self, root=root, arg_value=value, id_field=key, label=label)


_SHOPPING_GENERIC_READERS = {
    "read_seller_message": ("shopping_seller_messages", "message_id", "Seller message"),
    "read_support_followup_message": ("shopping_support_followup_messages", "message_id", "Support follow-up message"),
    "read_coupon_rule_note": ("shopping_coupon_rule_notes", "note_id", "Coupon rule note"),
    "read_shipping_policy_note": ("shopping_shipping_policy_notes", "note_id", "Shipping policy note"),
    "read_order_history_note": ("shopping_order_history_notes", "note_id", "Order history note"),
    "read_prior_return_note": ("shopping_prior_return_notes", "note_id", "Prior return note"),
    "read_fulfillment_history_note": ("shopping_fulfillment_history_notes", "note_id", "Fulfillment history note"),
    "read_invoice_or_receipt_text": ("shopping_invoice_or_receipt_texts", "attachment_id", "Invoice or receipt"),
    "read_packing_slip_text": ("shopping_packing_slip_texts", "attachment_id", "Packing slip"),
    "read_seller_attachment_text": ("shopping_seller_attachment_texts", "attachment_id", "Seller attachment"),
    "read_order_note": ("shopping_order_notes", "note_id", "Order note"),
    "read_return_agent_note": ("shopping_return_agent_notes", "note_id", "Return agent note"),
    "read_return_case_ticket": ("shopping_return_case_tickets", "ticket_id", "Return case ticket"),
    "read_order_support_ticket": ("shopping_order_support_tickets", "ticket_id", "Order support ticket"),
}


for _name, (_root, _key, _label) in _SHOPPING_GENERIC_READERS.items():
    def _factory(root: str, key: str, label: str):
        def _reader(self, **kwargs: Any) -> Dict[str, Any]:
            value = kwargs.get(key) or kwargs.get("id") or kwargs.get("note_id") or kwargs.get("message_id") or kwargs.get("attachment_id")
            return _shopping_v3_generic_reader(self, root, key, label, value)
        return _reader
    setattr(ShoppingTools, _name, _factory(_root, _key, _label))


ShoppingTools.remove_coupon = _shopping_v3_remove_coupon
ShoppingTools.remove_from_cart = _shopping_v3_remove_from_cart
ShoppingTools.get_return_eligibility = _shopping_v3_get_return_eligibility
ShoppingTools.get_coupon_eligibility = _shopping_v3_get_coupon_eligibility
ShoppingTools.get_cart_constraints = _shopping_v3_get_cart_constraints
ShoppingTools.get_delivery_window = _shopping_v3_get_delivery_window
ShoppingTools.get_seller_policy = _shopping_v3_get_seller_policy
ShoppingTools.get_restock_eta = _shopping_v3_get_restock_eta
