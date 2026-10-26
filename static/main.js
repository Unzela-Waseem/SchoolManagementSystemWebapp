
document.addEventListener("DOMContentLoaded", () => {
    // Dismiss flash messages.
    document.querySelectorAll(".alert-close").forEach((button) => {
        button.addEventListener("click", () => {
            button.closest(".alert")?.remove();
        });
    });

    // Confirm potentially disruptive account changes.
    document.querySelectorAll("form[data-confirm]").forEach((form) => {
        form.addEventListener("submit", (event) => {
            const message = form.dataset.confirm || "Are you sure?";
            if (!window.confirm(message)) {
                event.preventDefault();
            }
        });
    });

    // Client-side confirmation for matching passwords.
    const signupForm = document.querySelector('form[action*="signup"]');
    if (signupForm) {
        signupForm.addEventListener("submit", (event) => {
            const password = signupForm.querySelector('[name="password"]');
            const confirm = signupForm.querySelector('[name="confirm_password"]');

            if (password && confirm && password.value !== confirm.value) {
                event.preventDefault();
                alert("Passwords do not match.");
                confirm.focus();
            }
        });
    }
});
