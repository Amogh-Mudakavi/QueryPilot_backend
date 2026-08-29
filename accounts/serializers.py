from rest_framework import serializers
from rest_framework.exceptions import ValidationError
from .models import User , PendingRegistration



class RegisterRequestSerializer(serializers.Serializer):

    username = serializers.CharField(max_length = 100)
    email = serializers.EmailField()
    password = serializers.CharField(write_only = True)


    def validate_email(self , value):
        

        if User.objects.filter(email = value).exists():
            raise ValidationError('Email Already exists')

        if PendingRegistration.objects.filter(email = value).exists():
            raise ValidationError(
                "Registration already pending , Verify your email first"
            )
        
        return value
    
    def validate_username(self, value):

        if User.objects.filter(username=value).exists():
            raise ValidationError(
                "Username already exists"
            )

        return value
    

class VerifyOTPSerializer(serializers.Serializer):
    email = serializers.EmailField()
    otp  = serializers.CharField(max_length = 6)

class LoginSerializer(serializers.Serializer):

    email = serializers.EmailField()
    password = serializers.CharField(write_only = True)


    def validate(self , attrs):

        email = attrs.get('email')
        password = attrs.get('password')


        try:
            user = User.objects.get(email = email)
        
        except User.DoesNotExist:
            raise ValidationError("invalid credentials")
        
        if (user.check_password(password) == False):
            raise ValidationError('invalid credentials')
        
        attrs['user'] = user

        return attrs


class RefreshSerialier(serializers.Serializer):

    refresh_token = serializers.CharField()



class EmailSerializer(serializers.Serializer):

    email = serializers.EmailField()


class ResetPasswordSerializer(serializers.Serializer):

    email = serializers.EmailField()

    otp = serializers.CharField(max_length = 6)

    new_password = serializers.CharField(write_only = True)


    def validate_new_password(self , value):

        if len(value)<8:
            raise ValidationError(
                "Password must contain at least 8 characters"
            )
        
        return value


