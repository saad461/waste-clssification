import os
import asyncio
from playwright.async_api import async_playwright

async def capture_screenshots():
    # Use relative path for screenshots
    output_dir = "screenshots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch()

        # SXGA Resolution: 1280x1024
        context = await browser.new_context(
            viewport={'width': 1280, 'height': 1024},
            device_scale_factor=1.0
        )
        page = await context.new_page()

        # 1. Login Page
        print("Capturing Login Page...")
        await page.goto("http://127.0.0.1:5000/login")
        await page.screenshot(path=f"{output_dir}/login_page.jpg", type="jpeg", quality=100)

        # Login to get access to other pages
        await page.fill('input[name="username"]', 'testuser')
        await page.fill('input[name="password"]', 'password123')
        await page.click('button[type="submit"]')
        await page.wait_for_url("http://127.0.0.1:5000/dashboard")

        # 2. User Dashboard (Empty)
        print("Capturing User Dashboard...")
        await page.screenshot(path=f"{output_dir}/user_dashboard.jpg", type="jpeg", quality=100)

        # 3. Classifier Page (Upload form)
        print("Capturing Classifier Page...")
        await page.goto("http://127.0.0.1:5000/classifier")
        await page.screenshot(path=f"{output_dir}/classifier_page.jpg", type="jpeg", quality=100)

        # 4. Classify Image Result
        print("Capturing Classify Result...")
        # Upload a paper image
        # Assuming paper.jpg is in the current directory (we downloaded it earlier)
        await page.set_input_files('input[name="file"]', 'paper.jpg')
        await page.click('button[type="submit"]')
        # Wait for result to load
        await page.wait_for_selector('.result-container, .alert-success, img[src*="uploads"]')
        await page.screenshot(path=f"{output_dir}/classify_result.jpg", type="jpeg", quality=100)

        # 5. Admin Dashboard
        print("Capturing Admin Dashboard...")
        # First logout from user
        await page.goto("http://127.0.0.1:5000/logout")
        # Go to admin login
        await page.goto("http://127.0.0.1:5000/admin")
        await page.fill('input[name="username"]', 'admin')
        await page.fill('input[name="password"]', 'admin123')
        await page.click('button[type="submit"]')
        await page.wait_for_url("http://127.0.0.1:5000/admin/dashboard")
        await page.screenshot(path=f"{output_dir}/admin_dashboard.jpg", type="jpeg", quality=100)

        await browser.close()
        print(f"Screenshots captured successfully in {output_dir}/ directory.")

if __name__ == "__main__":
    asyncio.run(capture_screenshots())
