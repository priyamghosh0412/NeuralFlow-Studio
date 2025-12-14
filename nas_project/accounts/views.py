from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth.forms import AuthenticationForm

def login_view(request):
    if request.user.is_authenticated:
        return redirect('eda_view')
        
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('eda_view')
            else:
                messages.error(request, "Invalid username or password.")
        else:
            messages.error(request, "Invalid username or password.")
    else:
        form = AuthenticationForm()
        
    return render(request, 'accounts/login.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('login')

def forgot_password_view(request):
    if request.method == 'POST':
        user_id = request.POST.get('user_id')
        try:
            # Check if user exists by username (treating user_id as username for simplicity as per request)
            # Or if user meant numeric ID? Request said "id that can be an email or just any user specific id"
            # Let's assume Username first, as that's standard.
            user = User.objects.get(username=user_id)
            return render(request, 'accounts/forgot_password.html', {'step': 2, 'user_id': user.id, 'username': user.username})
        except User.DoesNotExist:
            messages.error(request, "Incorrect ID")
            return render(request, 'accounts/forgot_password.html', {'step': 1})
            
    return render(request, 'accounts/forgot_password.html', {'step': 1})

def reset_password_view(request, user_id):
    if request.method == 'POST':
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')
        
        if password != confirm_password:
            messages.error(request, "Passwords do not match")
            user = get_object_or_404(User, id=user_id)
            return render(request, 'accounts/forgot_password.html', {'step': 2, 'user_id': user.id, 'username': user.username})
            
        user = get_object_or_404(User, id=user_id)
        user.set_password(password)
        user.save()
        
        messages.success(request, "Password reset successfully. Please login.")
        return redirect('login')
        
    return redirect('forgot_password')

def register_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')
        
        if not username or not password:
            messages.error(request, "Please fill in all fields.")
            return render(request, 'accounts/register.html')
            
        if password != confirm_password:
            messages.error(request, "Passwords do not match.")
            return render(request, 'accounts/register.html')
            
        if User.objects.filter(username=username).exists():
            messages.error(request, "User ID already exists.")
            return render(request, 'accounts/register.html')
            
        User.objects.create_user(username=username, password=password)
        messages.success(request, "User created successfully. Please login.")
        return redirect('login')
        
    return render(request, 'accounts/register.html')
