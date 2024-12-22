const languageDropdown = document.getElementById('language-dropdown');

languageDropdown.addEventListener('change', function() {
    const selectedLanguage = languageDropdown.value;
    console.log('Selected language:', selectedLanguage); 
});
