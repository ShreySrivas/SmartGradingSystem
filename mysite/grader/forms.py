from django import forms

from .models import Question, Essay
class AnswerForm(forms.ModelForm):
    answer = forms.CharField(max_length=100000, widget=forms.Textarea(attrs={'rows': 5, 'placeholder': "What's on your mind?"}))
    file = forms.FileField(label='Attach a keywords list', help_text='max. 42 megabytes')
    class Meta:
        model = Essay
        fields = ['answer']