# src/scraper/web_scraper.py
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from pathlib import Path
import logging
import re

logger = logging.getLogger(__name__)

class WebScraper:
    """Scraping de conteúdo web"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def scrape_url(self, url):
        """
        Fazer scraping de uma URL
        
        Args:
            url: URL para scraper
        
        Returns:
            Dict com título, conteúdo e metadados
        """
        try:
            logger.info(f"Scrapendo: {url}")
            
            # Fazer request
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remover scripts e styles
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extrair título
            title = soup.find('h1')
            if title:
                title = title.get_text(strip=True)
            else:
                title = soup.find('title')
                if title:
                    title = title.get_text(strip=True)
                else:
                    title = url
            
            # Extrair conteúdo principal
            content = self._extract_content(soup)
            
            if not content:
                logger.warning(f"Nenhum conteúdo extraído de {url}")
                return None
            
            logger.info(f"✅ Scraping concluído: {len(content)} caracteres")
            
            return {
                'url': url,
                'title': title,
                'content': content,
                'scraped_at': datetime.now().isoformat(),
                'length': len(content)
            }
        
        except requests.exceptions.MissingSchema:
            logger.error(f"URL inválida: {url}")
            return None
        except requests.exceptions.ConnectionError:
            logger.error(f"Erro de conexão: {url}")
            return None
        except Exception as e:
            logger.error(f"Erro ao scraper {url}: {e}")
            return None
    
    def _extract_content(self, soup):
        """
        Extrair conteúdo principal (remover noise)
        """
        # Remover elementos desnecessários
        for element in soup(["nav", "footer", "header", "aside", "advertisement", "ads"]):
            element.decompose()
        
        # Tentar encontrar artigo principal
        article = soup.find(['article', 'main', 'div.content', 'div.article'])
        
        if not article:
            article = soup.body if soup.body else soup
        
        # Extrair texto
        text = article.get_text(separator='\n', strip=True)
        
        # Limpar espaços em branco extras
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    def save_to_document(self, scraped_data, documents_dir='data/documents'):
        """
        Salvar conteúdo scrapado como documento
        
        Args:
            scraped_data: Dict retornado por scrape_url()
            documents_dir: Diretório para salvar
        
        Returns:
            Caminho do arquivo salvo
        """
        if not scraped_data:
            return None
        
        # Criar nome do arquivo a partir da URL
        url = scraped_data['url']
        filename = self._create_filename(url)
        filepath = Path(documents_dir) / filename
        
        # Criar diretório se não existir
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Salvar conteúdo
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# {scraped_data['title']}\n\n")
            f.write(f"**Fonte:** {scraped_data['url']}\n")
            f.write(f"**Scrapado em:** {scraped_data['scraped_at']}\n\n")
            f.write("---\n\n")
            f.write(scraped_data['content'])
        
        logger.info(f"✅ Documento salvo: {filepath}")
        return str(filepath)
    
    def _create_filename(self, url):
        """
        Criar nome de arquivo a partir da URL
        """
        # Remover protocolo
        filename = url.replace('https://', '').replace('http://', '')
        # Remover www
        filename = filename.replace('www.', '')
        # Remover caracteres inválidos
        filename = re.sub(r'[^\w\-_.]', '_', filename)
        # Limitar tamanho
        filename = filename[:100]
        # Adicionar extensão
        if not filename.endswith('.txt'):
            filename += '.txt'
        
        return filename

# Teste
if __name__ == "__main__":
    scraper = WebScraper()
    
    # Testar com uma URL
    test_url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    
    print(f"Scrapando: {test_url}")
    data = scraper.scrape_url(test_url)
    
    if data:
        print(f"✅ Título: {data['title']}")
        print(f"✅ Caracteres: {data['length']}")
        filepath = scraper.save_to_document(data)
        print(f"✅ Salvo em: {filepath}")
    else:
        print("❌ Falha no scraping")