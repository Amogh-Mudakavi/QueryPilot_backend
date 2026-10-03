"""Migrations 0002 Alter Emailotp Otp And More module for QueryPilot."""

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("accounts", "0001_initial"),
    ]

    operations = [
        migrations.AlterField(
            model_name="emailotp",
            name="otp",
            field=models.CharField(max_length=128),
        ),
        migrations.AlterField(
            model_name="emailotp",
            name="verification_attempts",
            field=models.PositiveSmallIntegerField(default=0),
        ),
        migrations.AlterField(
            model_name="refreshtoken",
            name="token",
            field=models.CharField(max_length=64, unique=True),
        ),
    ]
